"""
Ultra-Fast 1:N Fingerprint Search System
=======================================

This module implements state-of-the-art scalable search algorithms for 1:N fingerprint matching:
1. Learned Indexing with deep neural networks
2. Product Quantization for memory efficiency
3. Hierarchical Navigable Small World (HNSW) for approximate search
4. GPU-accelerated FAISS integration
5. Multi-level caching and optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
import time
import pickle
import logging
from typing import List, Tuple, Dict, Optional, Union, Any
from dataclasses import dataclass
from collections import defaultdict, OrderedDict
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import sqlite3
import redis
import joblib
from sklearn.cluster import KMeans


@dataclass
class SearchResult:
    """Search result data structure"""
    template_id: str
    score: float
    distance: float
    rank: int
    metadata: Dict[str, Any] = None


@dataclass
class SearchConfig:
    """Configuration for search system"""
    # Index parameters
    index_type: str = 'HNSW'  # 'HNSW', 'IVF', 'LSH', 'Learned'
    dimension: int = 256
    
    # HNSW parameters
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100
    
    # IVF parameters
    ivf_ncentroids: int = 1024
    ivf_nprobe: int = 32
    
    # Quantization parameters
    use_pq: bool = False
    pq_m: int = 8
    pq_bits: int = 8
    
    # Search parameters
    k: int = 100
    similarity_threshold: float = 0.7
    max_candidates: int = 10000
    
    # Performance parameters
    batch_size: int = 32
    num_threads: int = 8
    use_gpu: bool = True
    gpu_ids: List[int] = None
    
    # Caching parameters
    use_cache: bool = True
    cache_size: int = 10000
    cache_ttl: int = 3600  # seconds


class LearnedIndex(nn.Module):
    """Neural network-based learned index for fingerprint templates"""
    
    def __init__(self, input_dim: int = 256, hidden_dims: List[int] = [512, 256, 128],
                 num_partitions: int = 1024):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_partitions = num_partitions
        
        # Neural network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer for partition prediction
        layers.append(nn.Linear(prev_dim, num_partitions))
        
        self.network = nn.Sequential(*layers)
        
        # Partition embeddings for refinement
        self.partition_embeddings = nn.Embedding(num_partitions, 64)
        
        # Refinement network
        self.refinement = nn.Sequential(
            nn.Linear(hidden_dims[-1] + 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass to predict partition and refinement score"""
        # Feature extraction
        features = x
        for layer in self.network[:-1]:
            features = layer(features)
        
        # Partition prediction
        partition_logits = self.network[-1](features)
        partition_probs = F.softmax(partition_logits, dim=1)
        
        # Get top partition
        top_partition = torch.argmax(partition_probs, dim=1)
        
        # Refinement using partition embeddings
        partition_emb = self.partition_embeddings(top_partition)
        refinement_input = torch.cat([features, partition_emb], dim=1)
        refinement_score = self.refinement(refinement_input)
        
        return {
            'partition_logits': partition_logits,
            'partition_probs': partition_probs,
            'predicted_partition': top_partition,
            'refinement_score': refinement_score.squeeze(1)
        }


class ProductQuantizer:
    """Product Quantization for memory-efficient storage"""
    
    def __init__(self, dimension: int, m: int = 8, bits: int = 8):
        self.dimension = dimension
        self.m = m
        self.bits = bits
        self.k = 2 ** bits  # Codebook size per subquantizer
        self.d_sub = dimension // m  # Dimension per subquantizer
        
        assert dimension % m == 0, "Dimension must be divisible by m"
        
        self.codebooks = None
        self.is_trained = False
        
    def train(self, X: np.ndarray) -> 'ProductQuantizer':
        """Train the product quantizer on data"""
        print(f"Training Product Quantizer with {len(X)} samples...")
        
        self.codebooks = []
        
        for i in range(self.m):
            start_idx = i * self.d_sub
            end_idx = (i + 1) * self.d_sub
            
            # Extract subvector
            X_sub = X[:, start_idx:end_idx]
            
            # Train k-means for this subquantizer
            kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=1)
            kmeans.fit(X_sub)
            
            self.codebooks.append(kmeans.cluster_centers_)
        
        self.codebooks = np.array(self.codebooks)  # [m, k, d_sub]
        self.is_trained = True
        
        return self
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode vectors using product quantization"""
        assert self.is_trained, "Quantizer must be trained first"
        
        N = X.shape[0]
        codes = np.zeros((N, self.m), dtype=np.uint8)
        
        for i in range(self.m):
            start_idx = i * self.d_sub
            end_idx = (i + 1) * self.d_sub
            
            X_sub = X[:, start_idx:end_idx]
            
            # Find closest centroid for each vector
            distances = np.linalg.norm(
                X_sub[:, None, :] - self.codebooks[i][None, :, :], 
                axis=2
            )
            codes[:, i] = np.argmin(distances, axis=1)
        
        return codes
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode quantized codes back to vectors"""
        assert self.is_trained, "Quantizer must be trained first"
        
        N = codes.shape[0]
        X_reconstructed = np.zeros((N, self.dimension))
        
        for i in range(self.m):
            start_idx = i * self.d_sub
            end_idx = (i + 1) * self.d_sub
            
            X_reconstructed[:, start_idx:end_idx] = self.codebooks[i][codes[:, i]]
        
        return X_reconstructed


class CacheManager:
    """Multi-level cache manager for search acceleration"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        
        # In-memory cache (LRU)
        self.memory_cache = OrderedDict()
        self.cache_stats = {'hits': 0, 'misses': 0}
        
        # Redis cache (optional)
        self.redis_client = None
        try:
            if config.use_cache:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self.redis_client.ping()  # Test connection
        except:
            logging.warning("Redis not available, using memory cache only")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        # Try memory cache first
        if key in self.memory_cache:
            # Move to end (LRU)
            self.memory_cache.move_to_end(key)
            self.cache_stats['hits'] += 1
            return self.memory_cache[key]
        
        # Try Redis cache
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    result = pickle.loads(data)
                    # Store in memory cache
                    self.put(key, result, update_redis=False)
                    self.cache_stats['hits'] += 1
                    return result
            except:
                pass
        
        self.cache_stats['misses'] += 1
        return None
    
    def put(self, key: str, value: Any, ttl: int = None, update_redis: bool = True) -> None:
        """Put item in cache"""
        # Memory cache
        self.memory_cache[key] = value
        
        # Enforce size limit
        while len(self.memory_cache) > self.config.cache_size:
            self.memory_cache.popitem(last=False)  # Remove oldest
        
        # Redis cache
        if self.redis_client and update_redis:
            try:
                ttl = ttl or self.config.cache_ttl
                data = pickle.dumps(value)
                self.redis_client.setex(key, ttl, data)
            except:
                pass


class UltraFastSearch:
    """Ultra-fast 1:N fingerprint search system"""
    
    def __init__(self, config: SearchConfig = None):
        if config is None:
            config = SearchConfig()
        
        self.config = config
        self.dimension = config.dimension
        
        # Initialize components
        self.cache_manager = CacheManager(config)
        self.product_quantizer = None
        self.learned_index = None
        
        # FAISS indices
        self.faiss_index = None
        self.faiss_gpu_resources = None
        
        # Template storage
        self.templates = {}  # template_id -> features
        self.template_metadata = {}  # template_id -> metadata
        self.template_id_to_index = {}  # template_id -> faiss_index
        
        # Statistics
        self.search_stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'avg_search_time': 0.0,
            'total_templates': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """Initialize the search index"""
        if self.config.index_type == 'HNSW':
            self._initialize_hnsw_index()
        elif self.config.index_type == 'IVF':
            self._initialize_ivf_index()
        elif self.config.index_type == 'Learned':
            self._initialize_learned_index()
        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")
    
    def _initialize_hnsw_index(self) -> None:
        """Initialize HNSW index"""
        if self.config.use_pq:
            # HNSW with Product Quantization
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.faiss_index = faiss.IndexPQ(
                self.dimension, self.config.pq_m, self.config.pq_bits
            )
        else:
            # Standard HNSW
            self.faiss_index = faiss.IndexHNSWFlat(self.dimension, self.config.hnsw_m)
            self.faiss_index.hnsw.efConstruction = self.config.hnsw_ef_construction
            self.faiss_index.hnsw.efSearch = self.config.hnsw_ef_search
        
        # GPU acceleration
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            self._setup_gpu_index()
    
    def _initialize_ivf_index(self) -> None:
        """Initialize IVF index"""
        quantizer = faiss.IndexFlatL2(self.dimension)
        
        if self.config.use_pq:
            self.faiss_index = faiss.IndexIVFPQ(
                quantizer, self.dimension, self.config.ivf_ncentroids,
                self.config.pq_m, self.config.pq_bits
            )
        else:
            self.faiss_index = faiss.IndexIVFFlat(
                quantizer, self.dimension, self.config.ivf_ncentroids
            )
        
        self.faiss_index.nprobe = self.config.ivf_nprobe
        
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            self._setup_gpu_index()
    
    def _setup_gpu_index(self) -> None:
        """Setup GPU acceleration for FAISS"""
        try:
            gpu_ids = self.config.gpu_ids or list(range(faiss.get_num_gpus()))
            
            if len(gpu_ids) == 1:
                # Single GPU
                res = faiss.StandardGpuResources()
                self.faiss_gpu_resources = res
                self.faiss_index = faiss.index_cpu_to_gpu(res, gpu_ids[0], self.faiss_index)
            else:
                # Multiple GPUs
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                co.useFloat16 = True
                
                self.faiss_index = faiss.index_cpu_to_all_gpus(self.faiss_index, co, gpu_ids)
            
            logging.info(f"GPU acceleration enabled on GPUs: {gpu_ids}")
        except Exception as e:
            logging.warning(f"Failed to setup GPU acceleration: {e}")
    
    def add_template(self, template_id: str, features: np.ndarray, 
                    metadata: Dict[str, Any] = None) -> None:
        """Add a template to the search index"""
        with self.lock:
            # Normalize features
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            features = features.astype(np.float32)
            
            # Store template
            current_index = len(self.templates)
            self.templates[template_id] = features[0]
            self.template_id_to_index[template_id] = current_index
            
            if metadata:
                self.template_metadata[template_id] = metadata
            
            # Add to FAISS index
            if self.faiss_index is not None:
                self.faiss_index.add(features)
            
            self.search_stats['total_templates'] += 1
    
    def search(self, query_features: np.ndarray, k: int = None, 
              similarity_threshold: float = None) -> List[SearchResult]:
        """Search for similar templates"""
        start_time = time.time()
        
        k = k or self.config.k
        similarity_threshold = similarity_threshold or self.config.similarity_threshold
        
        # Normalize query
        if len(query_features.shape) == 1:
            query_features = query_features.reshape(1, -1)
        query_features = query_features.astype(np.float32)
        
        # Check cache
        cache_key = f"search_{hash(query_features.tobytes())}_{k}_{similarity_threshold}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result is not None:
            self.search_stats['cache_hits'] += 1
            return cached_result
        
        # Perform search
        results = self._search_with_faiss(query_features, k, similarity_threshold)
        
        # Cache results
        self.cache_manager.put(cache_key, results)
        
        # Update statistics
        search_time = time.time() - start_time
        with self.lock:
            self.search_stats['total_searches'] += 1
            self.search_stats['avg_search_time'] = (
                (self.search_stats['avg_search_time'] * (self.search_stats['total_searches'] - 1) + 
                 search_time) / self.search_stats['total_searches']
            )
        
        return results
    
    def _search_with_faiss(self, query_features: np.ndarray, k: int, 
                          similarity_threshold: float) -> List[SearchResult]:
        """Search using FAISS index"""
        if self.faiss_index.ntotal == 0:
            return []
        
        # FAISS search
        distances, indices = self.faiss_index.search(query_features, min(k, self.faiss_index.ntotal))
        
        # Convert to results
        results = []
        template_ids = list(self.templates.keys())
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(template_ids):
                template_id = template_ids[idx]
                
                # Convert distance to similarity score
                score = 1.0 / (1.0 + distance)
                
                if score >= similarity_threshold:
                    results.append(SearchResult(
                        template_id=template_id,
                        score=score,
                        distance=distance,
                        rank=i + 1,
                        metadata=self.template_metadata.get(template_id)
                    ))
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search system statistics"""
        cache_stats = self.cache_manager.get_stats()
        
        return {
            'search_stats': self.search_stats,
            'cache_stats': cache_stats,
            'index_stats': {
                'total_vectors': self.faiss_index.ntotal if self.faiss_index else 0,
                'index_type': self.config.index_type,
                'dimension': self.dimension
            }
        }


class DistributedSearch:
    """Distributed search system for massive scale"""
    
    def __init__(self, num_shards: int = 4, config: SearchConfig = None):
        self.num_shards = num_shards
        self.config = config or SearchConfig()
        
        # Create shard indices
        self.shards = []
        for i in range(num_shards):
            shard_config = SearchConfig(**self.config.__dict__)
            shard_config.use_gpu = False  # Manage GPU allocation manually
            self.shards.append(UltraFastSearch(shard_config))
        
        # Load balancing
        self.current_shard = 0
        self.shard_lock = threading.Lock()
    
    def add_template(self, template_id: str, features: np.ndarray, 
                    metadata: Dict[str, Any] = None) -> None:
        """Add template to appropriate shard"""
        # Hash-based sharding
        shard_id = hash(template_id) % self.num_shards
        self.shards[shard_id].add_template(template_id, features, metadata)
    
    def search(self, query_features: np.ndarray, k: int = None,
              similarity_threshold: float = None) -> List[SearchResult]:
        """Distributed search across all shards"""
        k = k or self.config.k
        k_per_shard = k * 2  # Get more from each shard for better recall
        
        # Search all shards in parallel
        with ThreadPoolExecutor(max_workers=self.num_shards) as executor:
            futures = []
            for shard in self.shards:
                future = executor.submit(
                    shard.search, query_features, k_per_shard, similarity_threshold
                )
                futures.append(future)
            
            # Collect results from all shards
            all_results = []
            for future in futures:
                shard_results = future.result()
                all_results.extend(shard_results)
        
        # Merge and re-rank results
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(all_results[:k]):
            result.rank = i + 1
        
        return all_results[:k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get distributed system statistics"""
        shard_stats = []
        total_templates = 0
        total_searches = 0
        
        for i, shard in enumerate(self.shards):
            stats = shard.get_stats()
            shard_stats.append({f'shard_{i}': stats})
            total_templates += stats['search_stats']['total_templates']
            total_searches += stats['search_stats']['total_searches']
        
        return {
            'total_templates': total_templates,
            'total_searches': total_searches,
            'num_shards': self.num_shards,
            'shard_stats': shard_stats
        }


# Factory functions
def create_ultra_fast_search(config: SearchConfig = None) -> UltraFastSearch:
    """Create ultra-fast search system"""
    return UltraFastSearch(config)


def create_distributed_search(num_shards: int = 4, config: SearchConfig = None) -> DistributedSearch:
    """Create distributed search system"""
    return DistributedSearch(num_shards, config)


# Benchmarking utilities
class SearchBenchmark:
    """Benchmarking utilities for search systems"""
    
    def __init__(self, search_system):
        self.search_system = search_system
        
    def benchmark_throughput(self, num_queries: int = 1000, dimension: int = 256) -> Dict[str, float]:
        """Benchmark search throughput"""
        # Generate random query vectors
        queries = np.random.randn(num_queries, dimension).astype(np.float32)
        
        # Warm up
        for _ in range(10):
            self.search_system.search(queries[0])
        
        # Benchmark
        start_time = time.time()
        
        for query in queries:
            self.search_system.search(query)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = num_queries / total_time
        avg_latency = total_time / num_queries
        
        return {
            'throughput_qps': throughput,
            'avg_latency_ms': avg_latency * 1000,
            'total_time_seconds': total_time
        }
    
    def benchmark_accuracy(self, test_pairs: List[Tuple[np.ndarray, str]], 
                          ground_truth: Dict[str, List[str]]) -> Dict[str, float]:
        """Benchmark search accuracy"""
        total_queries = len(test_pairs)
        correct_top1 = 0
        correct_top10 = 0
        total_recall = 0.0
        
        for query_features, query_id in test_pairs:
            results = self.search_system.search(query_features, k=10)
            
            if query_id in ground_truth:
                gt_matches = set(ground_truth[query_id])
                
                # Top-1 accuracy
                if results and results[0].template_id in gt_matches:
                    correct_top1 += 1
                
                # Top-10 accuracy
                top10_ids = {r.template_id for r in results[:10]}
                if gt_matches & top10_ids:
                    correct_top10 += 1
                
                # Recall calculation
                if gt_matches:
                    found_matches = len(gt_matches & top10_ids)
                    recall = found_matches / len(gt_matches)
                    total_recall += recall
        
        return {
            'top1_accuracy': correct_top1 / total_queries,
            'top10_accuracy': correct_top10 / total_queries,
            'avg_recall': total_recall / total_queries
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create search system
    config = SearchConfig(
        index_type='HNSW',
        dimension=256,
        use_gpu=torch.cuda.is_available(),
        k=100
    )
    
    search_system = create_ultra_fast_search(config)
    
    # Add some dummy templates
    print("Adding templates...")
    for i in range(1000):
        template_id = f"template_{i:04d}"
        features = np.random.randn(256).astype(np.float32)
        metadata = {'person_id': f"person_{i//10}", 'finger': i % 10}
        
        search_system.add_template(template_id, features, metadata)
    
    print(f"Added {search_system.search_stats['total_templates']} templates")
    
    # Perform some searches
    print("\nPerforming searches...")
    for i in range(10):
        query = np.random.randn(256).astype(np.float32)
        results = search_system.search(query, k=5)
        
        print(f"Query {i+1}: Found {len(results)} results")
        for j, result in enumerate(results[:3]):
            print(f"  {j+1}. {result.template_id} (score: {result.score:.4f})")
    
    # Show statistics
    stats = search_system.get_stats()
    print(f"\nSystem Statistics:")
    print(f"Total searches: {stats['search_stats']['total_searches']}")
    print(f"Cache hit rate: {stats['cache_stats']['hit_rate']:.2%}")
    print(f"Average search time: {stats['search_stats']['avg_search_time']*1000:.2f}ms")
    
    # Benchmark
    print("\nRunning benchmark...")
    benchmark = SearchBenchmark(search_system)
    throughput_results = benchmark.benchmark_throughput(num_queries=100)
    
    print(f"Throughput: {throughput_results['throughput_qps']:.1f} QPS")
    print(f"Average latency: {throughput_results['avg_latency_ms']:.2f}ms")
