# M15: Distributed Computing

**Status:** Proposed
**Priority:** 🟡 MEDIUM
**Estimated Duration:** 8-10 weeks
**Dependencies:** M2 (performance baseline), M10 (I/O optimization), M14 (PDK for large-scale simulations)

## Problem Statement

SpiceLab currently runs simulations on a single machine, limiting the scale and speed of Monte Carlo analysis, parameter sweeps, and optimization tasks. Modern cloud infrastructure and distributed computing frameworks enable parallelization across hundreds or thousands of cores, but SpiceLab lacks integration with these systems.

### Current Gaps
- ❌ No distributed task execution (limited to local multiprocessing)
- ❌ No cloud deployment support (AWS, GCP, Azure)
- ❌ No job scheduling and queue management
- ❌ No auto-scaling based on workload
- ❌ No cost optimization for cloud compute
- ❌ No distributed result aggregation
- ❌ Limited to local machine resources (CPU, memory)

### Impact
- **Scale:** Cannot run 10k+ Monte Carlo iterations efficiently
- **Speed:** Large parameter sweeps take hours instead of minutes
- **Cost:** Users must maintain expensive local compute clusters
- **Accessibility:** Barrier to entry for large-scale simulations

## Objectives

1. **Dask integration** - Distributed arrays, task graphs, parallel execution
2. **Ray support** - Actor model for Monte Carlo, optimization, ML workflows
3. **Kubernetes operator** - Job scheduling, autoscaling, resource management
4. **AWS Batch runner** - EC2 Spot instances, S3 result storage
5. **GCP Cloud Run integration** - Serverless simulation execution
6. **Azure Container Instances** - Batch job processing
7. **Cost optimization** - Spot/preemptible VMs, budget controls
8. **Target:** 100x parallelization, cloud deployment guides, cost-effective scaling

## Technical Design

### 1. Distributed Computing Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  SpiceLab Distributed Layer                  │
├──────────────────────────────────────────────────────────────┤
│  Orchestrator (Local or Cloud)                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ Task Queue │  │ Scheduler  │  │ Aggregator │            │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘            │
│        │                │                │                   │
│  ┌─────┴────────────────┴────────────────┴──────┐           │
│  │          Execution Backend                    │           │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐     │           │
│  │  │ Dask │  │ Ray  │  │  K8s │  │ Cloud│     │           │
│  │  └──────┘  └──────┘  └──────┘  └──────┘     │           │
│  └───────────────────────────────────────────────┘           │
│                          ↓                                   │
│  Worker Nodes (Local or Cloud)                               │
│  ┌────────────┬────────────┬────────────┬────────────┐      │
│  │ Worker 1   │ Worker 2   │ Worker N   │ ...        │      │
│  │ • Circuit  │ • Circuit  │ • Circuit  │            │      │
│  │ • Engine   │ • Engine   │ • Engine   │            │      │
│  │ • Results  │ • Results  │ • Results  │            │      │
│  └────────────┴────────────┴────────────┴────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

### 2. Dask Integration

**Distributed Monte Carlo with Dask:**

```python
# spicelab/distributed/dask_backend.py
from dask.distributed import Client, as_completed
import dask.array as da
import xarray as xr
from typing import Callable

class DaskBackend:
    """Dask-based distributed execution backend."""

    def __init__(
        self,
        scheduler_address: str | None = None,
        n_workers: int = 4,
        threads_per_worker: int = 1
    ):
        if scheduler_address:
            self.client = Client(scheduler_address)
        else:
            # Local cluster
            self.client = Client(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker
            )

    async def run_monte_carlo(
        self,
        circuit_factory: Callable,
        n_iterations: int,
        chunk_size: int = 100
    ) -> xr.Dataset:
        """Run Monte Carlo analysis distributed across Dask workers."""

        # Split iterations into chunks
        chunks = [
            range(i, min(i + chunk_size, n_iterations))
            for i in range(0, n_iterations, chunk_size)
        ]

        # Submit tasks to Dask
        futures = []
        for chunk in chunks:
            future = self.client.submit(
                self._run_monte_carlo_chunk,
                circuit_factory,
                chunk
            )
            futures.append(future)

        # Gather results as they complete
        results = []
        for future in as_completed(futures):
            chunk_result = future.result()
            results.append(chunk_result)

        # Concatenate all chunk results
        combined = xr.concat(results, dim="iteration")
        return combined

    def _run_monte_carlo_chunk(
        self,
        circuit_factory: Callable,
        iteration_range: range
    ) -> xr.Dataset:
        """Run a chunk of Monte Carlo iterations (executed on worker)."""
        from spicelab import run_simulation  # Import on worker

        chunk_results = []
        for i in iteration_range:
            circuit = circuit_factory(iteration=i)
            result = run_simulation(circuit, analyses=["tran"])
            chunk_results.append(result)

        return xr.concat(chunk_results, dim="iteration")

    def run_parameter_sweep(
        self,
        circuit_factory: Callable,
        parameter_grid: dict[str, list]
    ) -> xr.Dataset:
        """Run parameter sweep distributed across workers."""
        import itertools

        # Generate all parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        combinations = list(itertools.product(*param_values))

        # Submit tasks
        futures = [
            self.client.submit(
                self._run_single_simulation,
                circuit_factory,
                dict(zip(param_names, combo))
            )
            for combo in combinations
        ]

        # Gather results
        results = self.client.gather(futures)

        # Combine into multi-dimensional xarray
        return self._combine_sweep_results(results, parameter_grid)

    def shutdown(self):
        """Shutdown Dask client and workers."""
        self.client.close()
```

### 3. Ray Integration

**Ray actors for long-running simulations:**

```python
# spicelab/distributed/ray_backend.py
import ray
from typing import Callable
import xarray as xr

@ray.remote
class SimulationWorker:
    """Ray actor for running simulations."""

    def __init__(self):
        # Initialize SPICE engine on actor startup
        from spicelab import NGSpiceEngine
        self.engine = NGSpiceEngine()

    def run_simulation(self, circuit_netlist: str, analysis: str) -> xr.Dataset:
        """Run single simulation on this worker."""
        # Deserialize circuit, run simulation, return results
        ...

class RayBackend:
    """Ray-based distributed execution backend."""

    def __init__(
        self,
        redis_address: str | None = None,
        num_cpus: int | None = None
    ):
        if not ray.is_initialized():
            ray.init(
                address=redis_address or "auto",
                num_cpus=num_cpus
            )

        # Create worker pool
        self.workers = [
            SimulationWorker.remote()
            for _ in range(num_cpus or ray.available_resources()["CPU"])
        ]

    async def run_monte_carlo(
        self,
        circuit_factory: Callable,
        n_iterations: int
    ) -> xr.Dataset:
        """Distribute Monte Carlo across Ray workers."""

        # Round-robin task assignment to workers
        futures = []
        for i in range(n_iterations):
            worker = self.workers[i % len(self.workers)]
            circuit = circuit_factory(iteration=i)
            netlist = circuit.to_netlist()

            future = worker.run_simulation.remote(netlist, "tran")
            futures.append(future)

        # Gather results
        results = ray.get(futures)

        return xr.concat(results, dim="iteration")

    def run_optimization(
        self,
        objective_fn: Callable,
        parameter_space: dict,
        n_trials: int = 100
    ):
        """Ray-based hyperparameter optimization."""
        from ray import tune

        # Define search space
        config = {
            name: tune.uniform(bounds[0], bounds[1])
            for name, bounds in parameter_space.items()
        }

        # Run optimization
        analysis = tune.run(
            objective_fn,
            config=config,
            num_samples=n_trials,
            resources_per_trial={"cpu": 1}
        )

        return analysis.best_config
```

### 4. Kubernetes Operator

**Deploy SpiceLab simulations on Kubernetes:**

```yaml
# kubernetes/spicelab-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: spicelab-monte-carlo
spec:
  parallelism: 10  # 10 parallel pods
  completions: 100  # 100 total simulations
  template:
    spec:
      containers:
      - name: spicelab-worker
        image: spicelab/worker:latest
        env:
        - name: CIRCUIT_CONFIG
          valueFrom:
            configMapKeyRef:
              name: circuit-config
              key: circuit.yaml
        - name: ITERATION
          value: "{{.Index}}"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
      restartPolicy: Never
  backoffLimit: 3
```

**Python Kubernetes client:**

```python
# spicelab/distributed/k8s_backend.py
from kubernetes import client, config
from kubernetes.client.rest import ApiException

class KubernetesBackend:
    """Kubernetes-based batch job execution."""

    def __init__(self, kubeconfig_path: str | None = None):
        if kubeconfig_path:
            config.load_kube_config(kubeconfig_path)
        else:
            config.load_incluster_config()  # For in-cluster execution

        self.batch_api = client.BatchV1Api()
        self.core_api = client.CoreV1Api()

    def submit_monte_carlo_job(
        self,
        circuit: 'Circuit',
        n_iterations: int,
        parallelism: int = 10,
        image: str = "spicelab/worker:latest"
    ) -> str:
        """Submit Monte Carlo job to Kubernetes."""

        # Create ConfigMap with circuit definition
        config_map = self._create_circuit_configmap(circuit)

        # Create Job
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=f"spicelab-mc-{circuit.name}",
                labels={"app": "spicelab", "circuit": circuit.name}
            ),
            spec=client.V1JobSpec(
                parallelism=parallelism,
                completions=n_iterations,
                template=client.V1PodTemplateSpec(
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="worker",
                                image=image,
                                env=[
                                    client.V1EnvVar(
                                        name="CIRCUIT_CONFIG",
                                        value_from=client.V1EnvVarSource(
                                            config_map_key_ref=client.V1ConfigMapKeySelector(
                                                name=config_map.metadata.name,
                                                key="circuit.yaml"
                                            )
                                        )
                                    )
                                ],
                                resources=client.V1ResourceRequirements(
                                    requests={"cpu": "1", "memory": "2Gi"},
                                    limits={"cpu": "2", "memory": "4Gi"}
                                )
                            )
                        ],
                        restart_policy="Never"
                    )
                )
            )
        )

        # Submit job
        response = self.batch_api.create_namespaced_job(
            namespace="default",
            body=job
        )

        return response.metadata.name

    def get_job_status(self, job_name: str) -> dict:
        """Get status of submitted job."""
        job = self.batch_api.read_namespaced_job(
            name=job_name,
            namespace="default"
        )

        return {
            "active": job.status.active or 0,
            "succeeded": job.status.succeeded or 0,
            "failed": job.status.failed or 0,
            "completion_time": job.status.completion_time
        }
```

### 5. AWS Batch Integration

```python
# spicelab/distributed/aws_backend.py
import boto3
from typing import Dict

class AWSBatchBackend:
    """AWS Batch execution backend."""

    def __init__(self, job_queue: str, job_definition: str):
        self.batch_client = boto3.client('batch')
        self.s3_client = boto3.client('s3')
        self.job_queue = job_queue
        self.job_definition = job_definition

    def submit_monte_carlo(
        self,
        circuit: 'Circuit',
        n_iterations: int,
        s3_bucket: str,
        instance_type: str = "c5.xlarge",
        use_spot: bool = True
    ) -> list[str]:
        """Submit Monte Carlo to AWS Batch."""

        # Upload circuit to S3
        circuit_key = f"circuits/{circuit.name}.yaml"
        self.s3_client.put_object(
            Bucket=s3_bucket,
            Key=circuit_key,
            Body=circuit.to_yaml()
        )

        # Submit array job
        job_ids = []
        for i in range(n_iterations):
            response = self.batch_client.submit_job(
                jobName=f"spicelab-{circuit.name}-{i}",
                jobQueue=self.job_queue,
                jobDefinition=self.job_definition,
                containerOverrides={
                    'environment': [
                        {'name': 'S3_BUCKET', 'value': s3_bucket},
                        {'name': 'CIRCUIT_KEY', 'value': circuit_key},
                        {'name': 'ITERATION', 'value': str(i)},
                        {'name': 'USE_SPOT', 'value': str(use_spot)}
                    ]
                }
            )
            job_ids.append(response['jobId'])

        return job_ids

    def monitor_jobs(self, job_ids: list[str]) -> dict:
        """Monitor job completion."""
        status_counts = {"RUNNING": 0, "SUCCEEDED": 0, "FAILED": 0}

        for job_id in job_ids:
            response = self.batch_client.describe_jobs(jobs=[job_id])
            status = response['jobs'][0]['status']
            status_counts[status] = status_counts.get(status, 0) + 1

        return status_counts

    def aggregate_results(self, s3_bucket: str, result_prefix: str) -> xr.Dataset:
        """Download and aggregate results from S3."""
        # List all result files
        response = self.s3_client.list_objects_v2(
            Bucket=s3_bucket,
            Prefix=result_prefix
        )

        results = []
        for obj in response.get('Contents', []):
            # Download result file
            result = self.s3_client.get_object(
                Bucket=s3_bucket,
                Key=obj['Key']
            )
            # Parse and append
            results.append(xr.open_dataset(result['Body']))

        return xr.concat(results, dim="iteration")
```

### 6. Cost Optimization

```python
# spicelab/distributed/cost_optimizer.py
from dataclasses import dataclass
from typing import Literal

@dataclass
class CostProfile:
    """Cloud cost configuration."""
    provider: Literal["aws", "gcp", "azure"]
    instance_type: str
    hourly_cost: float
    use_spot: bool = True
    spot_discount: float = 0.7  # 70% discount
    budget_limit: float | None = None  # USD

class CostOptimizer:
    """Optimize cloud cost for simulations."""

    def __init__(self, cost_profile: CostProfile):
        self.profile = cost_profile

    def estimate_cost(
        self,
        n_simulations: int,
        avg_sim_time: float,  # seconds
        parallelism: int
    ) -> dict:
        """Estimate total cost for simulation campaign."""

        # Calculate total compute time
        total_time_hours = (n_simulations * avg_sim_time) / (parallelism * 3600)

        # Apply spot discount if enabled
        effective_cost = self.profile.hourly_cost
        if self.profile.use_spot:
            effective_cost *= self.profile.spot_discount

        total_cost = total_time_hours * effective_cost * parallelism

        return {
            "total_cost_usd": total_cost,
            "compute_hours": total_time_hours,
            "instance_hours": total_time_hours * parallelism,
            "within_budget": (
                total_cost <= self.profile.budget_limit
                if self.profile.budget_limit else True
            )
        }

    def recommend_parallelism(
        self,
        n_simulations: int,
        budget: float,
        avg_sim_time: float
    ) -> int:
        """Recommend optimal parallelism given budget."""

        effective_cost = (
            self.profile.hourly_cost * self.profile.spot_discount
            if self.profile.use_spot
            else self.profile.hourly_cost
        )

        # Binary search for max parallelism within budget
        max_parallel = min(n_simulations, 1000)  # Cap at 1000
        for p in range(1, max_parallel + 1):
            cost_est = self.estimate_cost(n_simulations, avg_sim_time, p)
            if cost_est["total_cost_usd"] > budget:
                return p - 1

        return max_parallel
```

## Implementation Plan

### Phase 1: Dask Integration (Weeks 1-3)
- [ ] Implement DaskBackend class
- [ ] Distributed Monte Carlo
- [ ] Distributed parameter sweeps
- [ ] Result aggregation
- [ ] Local cluster testing
- [ ] Remote cluster setup guide

### Phase 2: Ray Integration (Weeks 4-5)
- [ ] Implement RayBackend class
- [ ] Actor-based simulation workers
- [ ] Ray Tune integration (optimization)
- [ ] Performance benchmarking

### Phase 3: Kubernetes (Weeks 6-7)
- [ ] Kubernetes job templates
- [ ] Python K8s client integration
- [ ] Job monitoring and logs
- [ ] Helm chart for SpiceLab
- [ ] Autoscaling configuration

### Phase 4: Cloud Providers (Weeks 8-9)
- [ ] AWS Batch integration
- [ ] GCP Cloud Run integration
- [ ] Azure Container Instances
- [ ] S3/GCS/Blob storage for results
- [ ] Cost estimation tools

### Phase 5: Optimization & Documentation (Week 10)
- [ ] Cost optimizer
- [ ] Spot instance handling
- [ ] Deployment guides (AWS, GCP, Azure)
- [ ] Performance comparison benchmarks
- [ ] Tutorials and examples

## Success Metrics

### Must Have
- [ ] **100x parallelization** for Monte Carlo (1000 iterations in <10 min)
- [ ] **Dask and Ray** backends functional
- [ ] **Kubernetes** deployment working
- [ ] **AWS Batch** integration complete
- [ ] **Cost estimation** tools available

### Should Have
- [ ] **GCP and Azure** integrations
- [ ] **Auto-scaling** based on workload
- [ ] **Spot instance** cost savings >60%
- [ ] **10+ deployment examples**

### Nice to Have
- [ ] On-premise cluster support (SLURM, PBS)
- [ ] Hybrid cloud orchestration
- [ ] Real-time cost monitoring dashboard

## Dependencies

- M2 (Performance) - baseline metrics
- M10 (I/O) - efficient result storage
- M14 (PDK) - for large-scale IC simulations

## References

- [Dask Documentation](https://docs.dask.org/)
- [Ray Documentation](https://docs.ray.io/)
- [Kubernetes Batch Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/job/)
- [AWS Batch](https://aws.amazon.com/batch/)
