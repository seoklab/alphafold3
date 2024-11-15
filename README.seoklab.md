# Seoklab AF3 README

## Installation

```bash
unset CC CXX CFLAGS CXXFLAGS LDFLAGS CUDA_HOME
mamba env create -f environment.yaml
conda activate af3

CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" pip install -v .
build_data
```

## Usage

```bash
$ af3 --helpfull

       USAGE: af3 [flags]

flags:
  --buckets: Strictly increasing order of token sizes for which to cache
    compilations. For any input with more tokens than the largest bucket size, a
    new bucket is created for exactly that number of tokens.
    (default: '256,512,768,1024,1280,1536,2048,2560,3072,3584,4096,4608,5120')
    (a comma separated list)
  --db_dir: Path to the directory containing the databases.
    (default: '/applic/AlphaFold3/public_databases')
  --flash_attention_implementation: <triton|cudnn|xla>: Flash attention
    implementation to use. 'triton' and 'cudnn' uses a Triton and cuDNN flash
    attention implementation, respectively. The Triton kernel is fastest and has
    been tested more thoroughly. The Triton and cuDNN kernels require Ampere
    GPUs or later. 'xla' uses an XLA attention implementation (no flash
    attention) and is portable across GPU devices.
    (default: 'triton')
  --hmmalign_binary_path: Path to the Hmmalign binary.
    (default: '/data/galaxy4/user/jnooree/tmp/env/bin/hmmalign')
  --hmmbuild_binary_path: Path to the Hmmbuild binary.
    (default: '/data/galaxy4/user/jnooree/tmp/env/bin/hmmbuild')
  --hmmsearch_binary_path: Path to the Hmmsearch binary.
    (default: '/data/galaxy4/user/jnooree/tmp/env/bin/hmmsearch')
  --input_dir: Path to the directory containing input JSON files.
  --jackhmmer_binary_path: Path to the Jackhmmer binary.
    (default: '/data/galaxy4/user/jnooree/tmp/env/bin/jackhmmer')
  --jackhmmer_n_cpu: Number of CPUs to use for Jackhmmer. Default to
    min(cpu_count, 8). Going beyond 8 CPUs provides very little additional
    speedup.
    (default: '8')
    (an integer)
  --jax_compilation_cache_dir: Path to a directory for the JAX compilation
    cache.
  --json_path: Path to the input JSON file.
  --mgnify_database_path: Mgnify database path, used for protein MSA search.
    (default: '${DB_DIR}/mgy_clusters_2022_05.fa')
  --model_dir: Path to the model to use for inference.
    (default: '/applic/AlphaFold3/models')
  --nhmmer_binary_path: Path to the Nhmmer binary.
    (default: '/data/galaxy4/user/jnooree/tmp/env/bin/nhmmer')
  --nhmmer_n_cpu: Number of CPUs to use for Nhmmer. Default to min(cpu_count,
    8). Going beyond 8 CPUs provides very little additional speedup.
    (default: '8')
    (an integer)
  --ntrna_database_path: NT-RNA database path, used for RNA MSA search.
    (default:
    '${DB_DIR}/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta')
  --output_dir: Path to a directory where the results will be saved.
  --pdb_database_path: PDB database directory with mmCIF files path, used for
    template search.
    (default: '${DB_DIR}/pdb_mmcif_files')
  --rfam_database_path: Rfam database path, used for RNA MSA search.
    (default: '${DB_DIR}/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta')
  --rna_central_database_path: RNAcentral database path, used for RNA MSA
    search.
    (default: '${DB_DIR}/rnacentral_active_seq_id_90_cov_80_linclust.fasta')
  --[no]run_data_pipeline: Whether to run the data pipeline on the fold inputs.
    (default: 'true')
  --[no]run_inference: Whether to run inference on the fold inputs.
    (default: 'true')
  --seqres_database_path: PDB sequence database path, used for template search.
    (default: '${DB_DIR}/pdb_seqres.fasta')
  --small_bfd_database_path: Small BFD database path, used for protein MSA
    search.
    (default: '${DB_DIR}/bfd-first_non_consensus_sequences.fasta')
  --uniprot_cluster_annot_database_path: UniProt database path, used for protein
    paired MSA search.
    (default: '${DB_DIR}/uniprot_all.fa')
  --uniref90_database_path: UniRef90 database path, used for MSA search. The MSA
    obtained by searching it is used to construct the profile for template
    search.
    (default: '${DB_DIR}/uniref90.fa')
```
