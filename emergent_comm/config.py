from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvConfig:
    # Number of distinct values per attribute (e.g. 8 colors, 8 shapes, 8 sizes)
    n_colors: int = 8
    n_shapes: int = 8
    n_sizes: int = 4
    # How many distractor objects the receiver chooses between
    n_distractors: int = 3  # + 1 target = 4 total candidates (random baseline = 0.25)
    # Random seed for reproducibility
    seed: int = 42


@dataclass
class EncoderConfig:
    # Character/byte-level encoder (bidirectional transformer per word)
    vocab_size: int = 256        # UTF-8 byte alphabet
    d_model: int = 64            # Character embedding dimension
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    max_word_len: int = 16       # Max bytes per word segment


@dataclass
class BackboneConfig:
    # Word-level causal transformer
    d_model: int = 128           # Word embedding dimension (projected from encoder)
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    max_seq_len: int = 32        # Max number of word segments per message


@dataclass
class DecoderConfig:
    # Character-level causal decoder
    vocab_size: int = 256
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    max_word_len: int = 16


@dataclass
class SegmentationConfig:
    # Entropy-adaptive segmentation
    # A boundary is declared when entropy falls below this threshold
    entropy_threshold: float = 1.5
    # Minimum number of bytes before a boundary can be declared
    min_segment_len: int = 1
    # Maximum bytes before a forced boundary
    max_segment_len: int = 16
    # Smoothing window for entropy signal (larger = smoother)
    smoothing_window: int = 3


@dataclass
class AgentConfig:
    # Max total bytes a sender can emit per message (fixed-length during training)
    max_message_len: int = 8
    # End-of-message byte token (we reserve byte value 0)
    eos_byte: int = 0
    # Word separator token (we reserve byte value 1)
    word_sep_byte: int = 1
    # Temperature for sender sampling during training
    temperature: float = 1.0
    # Whether to use straight-through estimator (True) or REINFORCE (False)
    use_straight_through: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 256
    n_steps: int = 20_000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    eval_every: int = 500
    # Entropy regularization coefficient (encourages diverse messages)
    entropy_coeff: float = 0.005
    # REINFORCE baseline decay
    baseline_decay: float = 0.95
    checkpoint_dir: str = "checkpoints"
    seed: int = 42


@dataclass
class EvalConfig:
    # Number of samples for compositionality metrics
    n_topsim_samples: int = 500
    # Max items used for vocabulary frequency analysis
    n_vocab_samples: int = 2000
    # Whether to log emergent word examples during eval
    log_examples: bool = True
    n_examples: int = 10


@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    experiment_name: str = "emergent_comm_baseline"
