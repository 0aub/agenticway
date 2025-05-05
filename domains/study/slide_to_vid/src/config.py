from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import os
import yaml


class ConfigError(RuntimeError):
    """Raised when the user-supplied configuration is invalid."""


@dataclass(slots=True)
class VoiceCfg:
    id: str


@dataclass(slots=True)
class AppConfig:
    # ─── LLM ──────────────────────────────────────────────────────────────
    provider: str = "google"
    model_name: str = "gemini-2.0-flash"
    google_api_key: str = ""

    # ─── TTS ────────────────────────────────────────────
    tts_engine: str = "dia"           # "dia" | "polly"
    tts_model: str = "nari-labs/Dia-1.6B"
    
    dia_voices: Dict[str, str] = field(default_factory=dict)

    aws_region: str = "eu-north-1"
    aws_access_key: str = ""
    aws_secret_key: str = ""
    polly_voices: Dict[str, VoiceCfg] = field(
        default_factory=lambda: {
            "s1": VoiceCfg(id="Ivy"),
            "s2": VoiceCfg(id="Brian"),
        }
    )

    # ─── Rendering ───────────────────────────────────────────────────────
    output_suffix: str = "explained"
    speech_speed: float = 0.8

    # ─── Paths / runtime ─────────────────────────────────────────────────
    exp_name: str = "exp"
    models_dir: Path = Path("/app/shared/models")
    scratch_dir: Path = Path("/tmp/slide_voice_agent")
    test_mode: bool = False
    files: List[Path] = field(default_factory=list)

    # internal
    _raw: Optional[dict] = None

    # ─── Helpers ─────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "AppConfig":
        raw = Path(yaml_path).read_text()
        
        # Replace env vars in the format ${VAR_NAME}
        raw = os.path.expandvars(raw)

        data = yaml.safe_load(raw)
        cfg = cls(**data)
        cfg._raw = data
        cfg.validate()
        return cfg

    def validate(self) -> None:
        # Normalize types
        self.models_dir = Path(self.models_dir)
        self.scratch_dir = Path(self.scratch_dir)
        self.files = [Path(f) for f in self.files]

        # Convert dicts to VoiceCfg objects (only needed if using Polly)
        if self.tts_engine.lower() == "polly":
            self.polly_voices = {
                k: v if isinstance(v, VoiceCfg) else VoiceCfg(id=v)
                for k, v in self.polly_voices.items()
            }

        # Fail fast on missing secrets
        if self.provider.lower() == "google" and not self.google_api_key:
            raise ConfigError("google_api_key is required for Google provider")

        if self.tts_engine.lower() == "polly":
            missing = [k for k in ("aws_access_key", "aws_secret_key") if not getattr(self, k)]
            if missing:
                raise ConfigError(f"AWS Polly needs {', '.join(missing)}")

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    def save_snapshot(self, dst: Path) -> None:
        """Serialize the *original* YAML (if available) or this object to YAML."""
        import yaml

        data = self._raw or self.__dict__
        with open(dst, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                canonical=False,
            )
