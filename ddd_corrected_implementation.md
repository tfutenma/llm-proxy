# LLM Proxy Microservice - Implementação DDD Corrigida

## 📋 Análise dos Problemas DDD Identificados

### ❌ **Problemas Encontrados:**

1. **Ausência de Value Objects** - Enums simples sem validação rica
2. **Entidades Anêmicas** - Pouca lógica de domínio
3. **Serviços de Domínio Incorretos** - Mistura com Application Services
4. **Linguagem Ubíqua Inconsistente** - Termos técnicos misturados
5. **Agregados Mal Definidos** - Limites de consistência confusos

## 🏗️ Estrutura DDD Corrigida

### **Nova Organização do Domain:**

```
domain/
├── 📂 aggregates/              # Agregados do domínio
├── 📂 entities/                # Entidades
├── 📂 value_objects/           # Value Objects
├── 📂 services/                # Serviços de Domínio
├── 📂 repositories/            # Interfaces de repositório
├── 📂 events/                  # Eventos de domínio
└── 📂 specifications/          # Especificações do domínio
```

## 💎 Value Objects Implementados

### domain/value_objects/provider_config.py
```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum
import re

class ProviderType(str, Enum):
    """Tipos de provedor com validação de domínio"""
    AZURE_OPENAI = "azure_openai"
    GOOGLE_VERTEX = "google_vertex"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"

    def __post_init__(self):
        if not isinstance(self.value, str) or not self.value.strip():
            raise ValueError("Provider type cannot be empty")

@dataclass(frozen=True)
class ProviderConfig:
    """Value Object para configuração de provedor"""

    provider_type: ProviderType
    endpoint: str
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None
    region: Optional[str] = None
    additional_params: Dict[str, Any] = None

    def __post_init__(self):
        """Validações ricas do domínio"""
        if not self.endpoint or not self.endpoint.strip():
            raise ValueError("Endpoint cannot be empty")

        if not self._is_valid_url(self.endpoint):
            raise ValueError("Endpoint must be a valid URL")

        if self.provider_type == ProviderType.AZURE_OPENAI:
            self._validate_azure_config()
        elif self.provider_type == ProviderType.GOOGLE_VERTEX:
            self._validate_google_config()

    def _is_valid_url(self, url: str) -> bool:
        """Valida formato de URL"""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return url_pattern.match(url) is not None

    def _validate_azure_config(self):
        """Validações específicas do Azure"""
        if not self.deployment_name:
            raise ValueError("Azure provider requires deployment_name")
        if not self.api_version:
            raise ValueError("Azure provider requires api_version")

        # Valida formato do deployment name
        if not re.match(r'^[a-zA-Z0-9-_]+$', self.deployment_name):
            raise ValueError("Invalid Azure deployment name format")

    def _validate_google_config(self):
        """Validações específicas do Google"""
        if not self.region:
            raise ValueError("Google provider requires region")

        # Valida formato da região
        if not re.match(r'^[a-z0-9-]+$', self.region):
            raise ValueError("Invalid Google region format")

    def get_connection_params(self) -> Dict[str, Any]:
        """Retorna parâmetros de conexão específicos do provedor"""
        base_params = {
            "endpoint": self.endpoint,
            "provider_type": self.provider_type.value
        }

        if self.api_version:
            base_params["api_version"] = self.api_version
        if self.deployment_name:
            base_params["deployment_name"] = self.deployment_name
        if self.region:
            base_params["region"] = self.region
        if self.additional_params:
            base_params.update(self.additional_params)

        return base_params

    def is_azure_provider(self) -> bool:
        """Verifica se é provedor Azure"""
        return self.provider_type == ProviderType.AZURE_OPENAI

    def is_google_provider(self) -> bool:
        """Verifica se é provedor Google"""
        return self.provider_type == ProviderType.GOOGLE_VERTEX

    def supports_streaming(self) -> bool:
        """Verifica se o provedor suporta streaming"""
        streaming_providers = {
            ProviderType.AZURE_OPENAI,
            ProviderType.OPENAI,
            ProviderType.ANTHROPIC
        }
        return self.provider_type in streaming_providers
```

### domain/value_objects/cost_structure.py
```python
from dataclasses import dataclass
from typing import Dict, Optional
from decimal import Decimal, ROUND_HALF_UP

@dataclass(frozen=True)
class TokenCost:
    """Value Object para custo por token"""

    cost_per_thousand: Decimal
    currency: str = "USD"

    def __post_init__(self):
        """Validações de domínio"""
        if self.cost_per_thousand < 0:
            raise ValueError("Token cost cannot be negative")

        if not isinstance(self.cost_per_thousand, Decimal):
            object.__setattr__(self, 'cost_per_thousand', Decimal(str(self.cost_per_thousand)))

        if not self.currency or len(self.currency) != 3:
            raise ValueError("Currency must be a 3-letter code (e.g., USD)")

    def calculate_cost(self, token_count: int) -> Decimal:
        """Calcula custo baseado no número de tokens"""
        if token_count < 0:
            raise ValueError("Token count cannot be negative")

        cost = (Decimal(token_count) / Decimal('1000')) * self.cost_per_thousand
        return cost.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)

@dataclass(frozen=True)
class CostStructure:
    """Value Object para estrutura completa de custos"""

    input_cost: TokenCost
    output_cost: TokenCost
    has_additional_fees: bool = False
    additional_fee_per_request: Optional[Decimal] = None

    def __post_init__(self):
        """Validações de domínio"""
        if self.has_additional_fees and self.additional_fee_per_request is None:
            raise ValueError("Additional fee must be specified when has_additional_fees is True")

        if self.additional_fee_per_request and self.additional_fee_per_request < 0:
            raise ValueError("Additional fee cannot be negative")

    def calculate_total_cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        """Calcula custo total da requisição"""
        input_cost = self.input_cost.calculate_cost(input_tokens)
        output_cost = self.output_cost.calculate_cost(output_tokens)

        total = input_cost + output_cost

        if self.has_additional_fees and self.additional_fee_per_request:
            total += self.additional_fee_per_request

        return total

    def get_cost_breakdown(self, input_tokens: int, output_tokens: int) -> Dict[str, Decimal]:
        """Retorna detalhamento dos custos"""
        breakdown = {
            "input_cost": self.input_cost.calculate_cost(input_tokens),
            "output_cost": self.output_cost.calculate_cost(output_tokens),
            "total_cost": self.calculate_total_cost(input_tokens, output_tokens)
        }

        if self.has_additional_fees:
            breakdown["additional_fee"] = self.additional_fee_per_request or Decimal('0')

        return breakdown

    def is_cost_effective_for_volume(self, monthly_token_estimate: int,
                                   threshold_cost: Decimal) -> bool:
        """Avalia se o modelo é economicamente viável para um volume mensal"""
        estimated_cost = self.calculate_total_cost(
            monthly_token_estimate // 2,  # Assume 50% input
            monthly_token_estimate // 2   # Assume 50% output
        )
        return estimated_cost <= threshold_cost
```

### domain/value_objects/operation_type.py
```python
from dataclasses import dataclass
from typing import Set, Dict, Any, Optional
from enum import Enum

class OperationCategory(str, Enum):
    """Categorias de operação com semântica de domínio"""
    TEXT_GENERATION = "text_generation"
    MULTIMODAL = "multimodal"
    AUDIO_PROCESSING = "audio_processing"
    EMBEDDINGS = "embeddings"
    FINE_TUNING = "fine_tuning"

class OperationComplexity(str, Enum):
    """Complexidade da operação"""
    SIMPLE = "simple"      # Operações básicas, rápidas
    MODERATE = "moderate"  # Operações com processamento médio
    COMPLEX = "complex"    # Operações que demandam muito processamento

@dataclass(frozen=True)
class OperationType:
    """Value Object para tipo de operação"""

    name: str
    category: OperationCategory
    complexity: OperationComplexity
    requires_streaming_support: bool = False
    requires_function_calling: bool = False
    requires_vision_support: bool = False
    requires_audio_support: bool = False
    max_context_tokens: Optional[int] = None
    estimated_processing_time_ms: Optional[int] = None

    def __post_init__(self):
        """Validações de domínio"""
        if not self.name or not self.name.strip():
            raise ValueError("Operation name cannot be empty")

        if len(self.name) > 64:
            raise ValueError("Operation name too long (max 64 characters)")

        if self.max_context_tokens and self.max_context_tokens <= 0:
            raise ValueError("Max context tokens must be positive")

        if self.estimated_processing_time_ms and self.estimated_processing_time_ms <= 0:
            raise ValueError("Processing time must be positive")

        self._validate_operation_consistency()

    def _validate_operation_consistency(self):
        """Valida consistência entre categoria e requisitos"""
        if self.category == OperationCategory.AUDIO_PROCESSING and not self.requires_audio_support:
            raise ValueError("Audio processing operations must require audio support")

        if self.category == OperationCategory.MULTIMODAL and not (
            self.requires_vision_support or self.requires_audio_support
        ):
            raise ValueError("Multimodal operations must require vision or audio support")

    def get_resource_requirements(self) -> Dict[str, Any]:
        """Retorna requisitos de recursos da operação"""
        return {
            "streaming": self.requires_streaming_support,
            "function_calling": self.requires_function_calling,
            "vision": self.requires_vision_support,
            "audio": self.requires_audio_support,
            "max_context": self.max_context_tokens,
            "complexity": self.complexity.value
        }

    def is_compatible_with_capabilities(self, model_capabilities: Set[str]) -> bool:
        """Verifica se a operação é compatível com as capacidades do modelo"""
        required_capabilities = set()

        if self.requires_streaming_support:
            required_capabilities.add("streaming")
        if self.requires_function_calling:
            required_capabilities.add("function_calling")
        if self.requires_vision_support:
            required_capabilities.add("vision")
        if self.requires_audio_support:
            required_capabilities.add("audio")

        return required_capabilities.issubset(model_capabilities)

    def estimate_cost_multiplier(self) -> float:
        """Estima multiplicador de custo baseado na complexidade"""
        multipliers = {
            OperationComplexity.SIMPLE: 1.0,
            OperationComplexity.MODERATE: 1.5,
            OperationComplexity.COMPLEX: 2.0
        }
        return multipliers[self.complexity]

    def is_real_time_operation(self) -> bool:
        """Verifica se é uma operação de tempo real"""
        real_time_threshold_ms = 1000  # 1 segundo
        return (self.estimated_processing_time_ms and
                self.estimated_processing_time_ms <= real_time_threshold_ms)

# Operações predefinidas do domínio
@dataclass(frozen=True)
class StandardOperations:
    """Operações padrão do domínio LLM"""

    CHAT_COMPLETION = OperationType(
        name="chat_completion",
        category=OperationCategory.TEXT_GENERATION,
        complexity=OperationComplexity.MODERATE,
        requires_streaming_support=True,
        max_context_tokens=32000,
        estimated_processing_time_ms=2000
    )

    TEXT_GENERATION = OperationType(
        name="text_generation",
        category=OperationCategory.TEXT_GENERATION,
        complexity=OperationComplexity.SIMPLE,
        requires_streaming_support=True,
        max_context_tokens=8000,
        estimated_processing_time_ms=1500
    )

    EMBEDDINGS = OperationType(
        name="embeddings",
        category=OperationCategory.EMBEDDINGS,
        complexity=OperationComplexity.SIMPLE,
        max_context_tokens=8192,
        estimated_processing_time_ms=500
    )

    IMAGE_ANALYSIS = OperationType(
        name="image_analysis",
        category=OperationCategory.MULTIMODAL,
        complexity=OperationComplexity.COMPLEX,
        requires_vision_support=True,
        max_context_tokens=16000,
        estimated_processing_time_ms=3000
    )

    AUDIO_TRANSCRIPTION = OperationType(
        name="audio_transcription",
        category=OperationCategory.AUDIO_PROCESSING,
        complexity=OperationComplexity.MODERATE,
        requires_audio_support=True,
        estimated_processing_time_ms=2500
    )

    FUNCTION_CALLING = OperationType(
        name="function_calling",
        category=OperationCategory.TEXT_GENERATION,
        complexity=OperationComplexity.COMPLEX,
        requires_function_calling=True,
        requires_streaming_support=True,
        max_context_tokens=32000,
        estimated_processing_time_ms=2500
    )

    @classmethod
    def get_all_operations(cls) -> Dict[str, OperationType]:
        """Retorna todas as operações padrão"""
        return {
            "chat_completion": cls.CHAT_COMPLETION,
            "text_generation": cls.TEXT_GENERATION,
            "embeddings": cls.EMBEDDINGS,
            "image_analysis": cls.IMAGE_ANALYSIS,
            "audio_transcription": cls.AUDIO_TRANSCRIPTION,
            "function_calling": cls.FUNCTION_CALLING
        }

    @classmethod
    def get_by_category(cls, category: OperationCategory) -> Dict[str, OperationType]:
        """Retorna operações filtradas por categoria"""
        all_ops = cls.get_all_operations()
        return {name: op for name, op in all_ops.items() if op.category == category}
```

### domain/value_objects/model_metadata.py
```python
from dataclasses import dataclass
from typing import Dict, Any, Set, Optional, List
from datetime import datetime
from enum import Enum

class ModelCapability(str, Enum):
    """Capacidades específicas do modelo"""
    STREAMING = "streaming"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    AUDIO = "audio"
    CODE_GENERATION = "code_generation"
    MULTILINGUAL = "multilingual"
    REASONING = "reasoning"
    MATH = "math"

class PerformanceTier(str, Enum):
    """Nível de performance do modelo"""
    FAST = "fast"          # < 2s resposta típica
    STANDARD = "standard"  # 2-5s resposta típica
    POWERFUL = "powerful"  # > 5s resposta típica, mas alta qualidade

@dataclass(frozen=True)
class ModelMetadata:
    """Value Object para metadados ricos do modelo"""

    version: str
    release_date: datetime
    context_window: int
    max_output_tokens: int
    capabilities: Set[ModelCapability]
    performance_tier: PerformanceTier
    training_data_cutoff: Optional[datetime] = None
    languages_supported: Set[str] = None
    quality_score: Optional[float] = None  # 0-100
    safety_rating: Optional[str] = None
    benchmark_scores: Dict[str, float] = None

    def __post_init__(self):
        """Validações de domínio"""
        if not self.version or not self.version.strip():
            raise ValueError("Model version cannot be empty")

        if self.context_window <= 0:
            raise ValueError("Context window must be positive")

        if self.max_output_tokens <= 0:
            raise ValueError("Max output tokens must be positive")

        if self.max_output_tokens > self.context_window:
            raise ValueError("Max output tokens cannot exceed context window")

        if self.quality_score is not None and not (0 <= self.quality_score <= 100):
            raise ValueError("Quality score must be between 0 and 100")

        if not self.capabilities:
            object.__setattr__(self, 'capabilities', set())

        if not self.languages_supported:
            object.__setattr__(self, 'languages_supported', {"en"})  # Default English

        if not self.benchmark_scores:
            object.__setattr__(self, 'benchmark_scores', {})

    def supports_capability(self, capability: ModelCapability) -> bool:
        """Verifica se o modelo suporta uma capacidade específica"""
        return capability in self.capabilities

    def supports_language(self, language_code: str) -> bool:
        """Verifica se o modelo suporta um idioma específico"""
        return language_code.lower() in {lang.lower() for lang in self.languages_supported}

    def is_suitable_for_context_size(self, required_context: int,
                                   output_buffer: int = 1000) -> bool:
        """Verifica se o modelo pode lidar com o tamanho de contexto necessário"""
        available_context = self.context_window - output_buffer
        return available_context >= required_context

    def get_benchmark_score(self, benchmark_name: str) -> Optional[float]:
        """Obtém pontuação de um benchmark específico"""
        return self.benchmark_scores.get(benchmark_name)

    def is_high_performance(self) -> bool:
        """Verifica se é um modelo de alta performance"""
        return (self.performance_tier == PerformanceTier.POWERFUL and
                self.quality_score and self.quality_score >= 80)

    def is_suitable_for_production(self) -> bool:
        """Verifica se o modelo é adequado para produção"""
        production_criteria = [
            self.quality_score and self.quality_score >= 70,
            self.safety_rating in ["high", "very_high", None],  # None assume seguro
            self.context_window >= 4000,  # Contexto mínimo para produção
        ]
        return all(criterion for criterion in production_criteria if criterion is not None)

    def get_recommended_use_cases(self) -> List[str]:
        """Sugere casos de uso baseado nas capacidades"""
        use_cases = []

        if ModelCapability.CODE_GENERATION in self.capabilities:
            use_cases.append("code_generation")

        if ModelCapability.REASONING in self.capabilities:
            use_cases.append("complex_reasoning")

        if ModelCapability.VISION in self.capabilities:
            use_cases.append("multimodal_analysis")

        if ModelCapability.FUNCTION_CALLING in self.capabilities:
            use_cases.append("api_integration")

        if self.performance_tier == PerformanceTier.FAST:
            use_cases.append("real_time_applications")

        if len(self.languages_supported) > 5:
            use_cases.append("multilingual_applications")

        return use_cases

    def compare_with(self, other: 'ModelMetadata') -> Dict[str, str]:
        """Compara com outro modelo retornando diferenças"""
        comparison = {}

        if self.context_window != other.context_window:
            comparison["context_window"] = (
                "larger" if self.context_window > other.context_window else "smaller"
            )

        if self.performance_tier != other.performance_tier:
            comparison["performance"] = f"this_is_{self.performance_tier.value}_other_is_{other.performance_tier.value}"

        capability_diff = self.capabilities.symmetric_difference(other.capabilities)
        if capability_diff:
            comparison["capabilities"] = f"different_capabilities: {capability_diff}"

        if self.quality_score and other.quality_score:
            if abs(self.quality_score - other.quality_score) > 5:
                comparison["quality"] = (
                    "higher" if self.quality_score > other.quality_score else "lower"
                )

        return comparison
```

## 🎯 Entidades Corrigidas

### domain/entities/llm_model.py
```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Optional
from datetime import datetime
import uuid

from domain.value_objects.provider_config import ProviderConfig
from domain.value_objects.cost_structure import CostStructure
from domain.value_objects.operation_type import OperationType, StandardOperations
from domain.value_objects.model_metadata import ModelMetadata, ModelCapability
from domain.events.model_events import ModelCreated, ModelUpdated, ModelStatusChanged

@dataclass
class LLMModel:
    """Entidade rica de modelo LLM com comportamentos de domínio"""

    # Identificadores
    id: str
    name: str
    code: str

    # Value Objects
    provider_config: ProviderConfig
    cost_structure: CostStructure
    metadata: ModelMetadata

    # Dados de negócio
    supported_operations: Set[OperationType] = field(default_factory=set)
    project_access: Set[str] = field(default_factory=set)
    is_private: bool = False
    is_active: bool = True

    # Auditoria
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None

    # Eventos de domínio
    _domain_events: List[Any] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Validações e inicializações pós-criação"""
        self._validate_business_rules()
        if not hasattr(self, '_is_new_entity'):
            self._mark_as_created()

    def _validate_business_rules(self):
        """Validações ricas de regras de negócio"""
        if not self.id or not self.id.strip():
            raise ValueError("Model ID is required and cannot be empty")

        if not self.name or len(self.name.strip()) < 3:
            raise ValueError("Model name must have at least 3 characters")

        if not self.code or not self.code.strip():
            raise ValueError("Model code is required")

        # Regra de negócio: modelos privados devem ter projetos associados
        if self.is_private and not self.project_access:
            raise ValueError("Private models must have at least one associated project")

        # Regra de negócio: modelos ativos devem ter pelo menos uma operação
        if self.is_active and not self.supported_operations:
            raise ValueError("Active models must support at least one operation")

    def _mark_as_created(self):
        """Marca modelo como criado e adiciona evento"""
        self._domain_events.append(
            ModelCreated(
                model_id=self.id,
                model_name=self.name,
                provider_type=self.provider_config.provider_type.value,
                created_at=self.created_at,
                created_by=self.created_by
            )
        )

    # Comportamentos ricos de domínio

    def activate(self, activated_by: str) -> None:
        """Ativa o modelo (regra de negócio)"""
        if self.is_active:
            raise ValueError("Model is already active")

        if not self.supported_operations:
            raise ValueError("Cannot activate model without supported operations")

        self.is_active = True
        self.updated_at = datetime.utcnow()

        self._domain_events.append(
            ModelStatusChanged(
                model_id=self.id,
                old_status="inactive",
                new_status="active",
                changed_by=activated_by,
                changed_at=self.updated_at
            )
        )

    def deactivate(self, deactivated_by: str, reason: str) -> None:
        """Desativa o modelo"""
        if not self.is_active:
            raise ValueError("Model is already inactive")

        self.is_active = False
        self.updated_at = datetime.utcnow()

        self._domain_events.append(
            ModelStatusChanged(
                model_id=self.id,
                old_status="active",
                new_status="inactive",
                changed_by=deactivated_by,
                changed_at=self.updated_at,
                reason=reason
            )
        )

    def add_operation_support(self, operation: OperationType) -> None:
        """Adiciona suporte a uma operação"""
        if not self._can_support_operation(operation):
            raise ValueError(f"Model cannot support operation {operation.name} due to capability constraints")

        self.supported_operations.add(operation)
        self.updated_at = datetime.utcnow()

    def remove_operation_support(self, operation: OperationType) -> None:
        """Remove suporte a uma operação"""
        if operation not in self.supported_operations:
            raise ValueError(f"Model does not support operation {operation.name}")

        self.supported_operations.remove(operation)

        # Regra de negócio: não pode ficar sem operações se estiver ativo
        if self.is_active and not self.supported_operations:
            raise ValueError("Cannot remove last operation from active model")

        self.updated_at = datetime.utcnow()

    def grant_project_access(self, project_id: str) -> None:
        """Concede acesso a um projeto"""
        if not project_id or not project_id.strip():
            raise ValueError("Project ID cannot be empty")

        self.project_access.add(project_id)
        self.updated_at = datetime.utcnow()

    def revoke_project_access(self, project_id: str) -> None:
        """Revoga acesso de um projeto"""
        if project_id not in self.project_access:
            raise ValueError(f"Project {project_id} does not have access to this model")

        self.project_access.remove(project_id)

        # Regra de negócio: modelos privados devem ter pelo menos um projeto
        if self.is_private and not self.project_access:
            raise ValueError("Cannot revoke last project access from private model")

        self.updated_at = datetime.utcnow()

    def calculate_operation_cost(self, operation: OperationType,
                               input_tokens: int, output_tokens: int) -> float:
        """Calcula custo de uma operação específica"""
        if operation not in self.supported_operations:
            raise ValueError(f"Model does not support operation {operation.name}")

        base_cost = self.cost_structure.calculate_total_cost(input_tokens, output_tokens)
        operation_multiplier = operation.estimate_cost_multiplier()

        return float(base_cost * operation_multiplier)

    def is_suitable_for_project(self, project_id: str) -> bool:
        """Verifica se o modelo é adequado para um projeto"""
        if not self.is_active:
            return False

        if self.is_private:
            return project_id in self.project_access

        return True

    def can_handle_request(self, operation: OperationType,
                          context_size: int, output_size: int = 1000) -> bool:
        """Verifica se o modelo pode processar uma requisição"""
        if not self.is_active:
            return False

        if operation not in self.supported_operations:
            return False

        return self.metadata.is_suitable_for_context_size(context_size, output_size)

    def get_performance_profile(self) -> Dict[str, Any]:
        """Retorna perfil de performance do modelo"""
        return {
            "performance_tier": self.metadata.performance_tier.value,
            "context_window": self.metadata.context_window,
            "capabilities": [cap.value for cap in self.metadata.capabilities],
            "supported_operations": [op.name for op in self.supported_operations],
            "estimated_cost_range": self._estimate_cost_range(),
            "quality_score": self.metadata.quality_score,
            "is_production_ready": self.metadata.is_suitable_for_production()
        }

    def _can_support_operation(self, operation: OperationType) -> bool:
        """Verifica se o modelo pode suportar uma operação baseado em suas capacidades"""
        model_capabilities = {cap.value for cap in self.metadata.capabilities}
        return operation.is_compatible_with_capabilities(model_capabilities)

    def _estimate_cost_range(self) -> Dict[str, float]:
        """Estima faixa de custos típicos"""
        # Simulação com tokens típicos
        low_cost = float(self.cost_structure.calculate_total_cost(100, 50))
        medium_cost = float(self.cost_structure.calculate_total_cost(1000, 500))
        high_cost = float(self.cost_structure.calculate_total_cost(5000, 2000))

        return {
            "low_usage": low_cost,
            "medium_usage": medium_cost,
            "high_usage": high_cost
        }

    def update_metadata(self, new_metadata: ModelMetadata, updated_by: str) -> None:
        """Atualiza metadados do modelo"""
        old_metadata = self.metadata
        self.metadata = new_metadata
        self.updated_at = datetime.utcnow()

        self._domain_events.append(
            ModelUpdated(
                model_id=self.id,
                updated_fields=["metadata"],
                updated_by=updated_by,
                updated_at=self.updated_at,
                old_values={"metadata": old_metadata},
                new_values={"metadata": new_metadata}
            )
        )

    def get_domain_events(self) -> List[Any]:
        """Retorna eventos de domínio"""
        return self._domain_events.copy()

    def clear_domain_events(self) -> None:
        """Limpa eventos de domínio"""
        self._domain_events.clear()

    def __eq__(self, other):
        """Igualdade baseada na identidade do domínio"""
        if not isinstance(other, LLMModel):
            return False
        return self.id == other.id

    def __hash__(self):
        """Hash baseado na identidade"""
        return hash(self.id)

    def __repr__(self):
        return (f"LLMModel(id={self.id}, name={self.name}, "
                f"provider={self.provider_config.provider_type.value}, "
                f"active={self.is_active})")
```

## 🛠️ Serviços de Domínio Corrigidos

### domain/services/model_compatibility_service.py
```python
from typing import List, Set, Dict, Any
from domain.entities.llm_model import LLMModel
from domain.value_objects.operation_type import OperationType
from domain.value_objects.model_metadata import ModelCapability

class ModelCompatibilityService:
    """Serviço de domínio para lógica de compatibilidade entre modelos e operações"""

    def find_compatible_models(self, models: List[LLMModel],
                             operation: OperationType,
                             context_size: int,
                             project_id: str = None) -> List[LLMModel]:
        """Encontra modelos compatíveis com uma operação e contexto"""
        compatible_models = []

        for model in models:
            if self._is_model_compatible(model, operation, context_size, project_id):
                compatible_models.append(model)

        # Ordena por adequação (performance tier e quality score)
        return self._sort_by_suitability(compatible_models, operation)

    def _is_model_compatible(self, model: LLMModel, operation: OperationType,
                           context_size: int, project_id: str = None) -> bool:
        """Verifica compatibilidade completa do modelo"""
        # Verifica status básico
        if not model.is_active:
            return False

        # Verifica acesso ao projeto
        if project_id and not model.is_suitable_for_project(project_id):
            return False

        # Verifica se suporta a operação
        if not model.can_handle_request(operation, context_size):
            return False

        return True

    def _sort_by_suitability(self, models: List[LLMModel],
                           operation: OperationType) -> List[LLMModel]:
        """Ordena modelos por adequação à operação"""
        def suitability_score(model: LLMModel) -> float:
            score = 0.0

            # Performance tier
            tier_scores = {"fast": 3.0, "standard": 2.0, "powerful": 1.0}
            score += tier_scores.get(model.metadata.performance_tier.value, 0)

            # Quality score
            if model.metadata.quality_score:
                score += model.metadata.quality_score / 100.0

            # Capability match bonus
            required_caps = self._get_operation_requirements(operation)
            model_caps = {cap.value for cap in model.metadata.capabilities}

            if required_caps.issubset(model_caps):
                score += 1.0

            return score

        return sorted(models, key=suitability_score, reverse=True)

    def _get_operation_requirements(self, operation: OperationType) -> Set[str]:
        """Extrai requisitos da operação"""
        requirements = set()

        if operation.requires_streaming_support:
            requirements.add("streaming")
        if operation.requires_function_calling:
            requirements.add("function_calling")
        if operation.requires_vision_support:
            requirements.add("vision")
        if operation.requires_audio_support:
            requirements.add("audio")

        return requirements

    def assess_migration_compatibility(self, source_model: LLMModel,
                                     target_model: LLMModel) -> Dict[str, Any]:
        """Avalia compatibilidade para migração entre modelos"""
        assessment = {
            "is_compatible": True,
            "compatibility_score": 0.0,
            "issues": [],
            "recommendations": []
        }

        # Compara operações suportadas
        source_ops = {op.name for op in source_model.supported_operations}
        target_ops = {op.name for op in target_model.supported_operations}

        missing_operations = source_ops - target_ops
        if missing_operations:
            assessment["is_compatible"] = False
            assessment["issues"].append(f"Target model missing operations: {missing_operations}")

        # Compara contexto
        if target_model.metadata.context_window < source_model.metadata.context_window:
            assessment["issues"].append("Target model has smaller context window")
            assessment["compatibility_score"] -= 0.3

        # Compara capacidades
        source_caps = source_model.metadata.capabilities
        target_caps = target_model.metadata.capabilities

        missing_capabilities = source_caps - target_caps
        if missing_capabilities:
            assessment["issues"].append(f"Target model missing capabilities: {missing_capabilities}")
            assessment["compatibility_score"] -= 0.2

        # Avalia qualidade
        if (source_model.metadata.quality_score and target_model.metadata.quality_score and
            target_model.metadata.quality_score < source_model.metadata.quality_score - 10):
            assessment["recommendations"].append("Target model has significantly lower quality score")
            assessment["compatibility_score"] -= 0.1

        # Calcula score final
        base_score = 1.0
        assessment["compatibility_score"] = max(0.0, base_score + assessment["compatibility_score"])

        if assessment["compatibility_score"] < 0.7:
            assessment["is_compatible"] = False

        return assessment
```

### domain/services/cost_calculation_service.py
```python
from typing import Dict, List, Any
from decimal import Decimal
from domain.entities.llm_model import LLMModel
from domain.value_objects.operation_type import OperationType

class CostCalculationService:
    """Serviço de domínio para cálculos avançados de custo"""

    def calculate_batch_operation_cost(self, model: LLMModel,
                                     operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcula custo para um lote de operações"""
        total_cost = Decimal('0')
        operation_costs = []
        total_input_tokens = 0
        total_output_tokens = 0

        for op_data in operations:
            operation_type = op_data['operation_type']
            input_tokens = op_data['input_tokens']
            output_tokens = op_data['output_tokens']

            if operation_type not in model.supported_operations:
                raise ValueError(f"Model {model.id} does not support operation {operation_type.name}")

            op_cost = Decimal(str(model.calculate_operation_cost(
                operation_type, input_tokens, output_tokens
            )))

            operation_costs.append({
                'operation': operation_type.name,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'cost': float(op_cost)
            })

            total_cost += op_cost
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

        # Aplica descontos por volume se aplicável
        volume_discount = self._calculate_volume_discount(total_input_tokens + total_output_tokens)
        discounted_cost = total_cost * (1 - volume_discount)

        return {
            'total_cost': float(discounted_cost),
            'original_cost': float(total_cost),
            'volume_discount': volume_discount,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'operation_breakdown': operation_costs
        }

    def _calculate_volume_discount(self, total_tokens: int) -> Decimal:
        """Calcula desconto por volume baseado no total de tokens"""
        if total_tokens > 1000000:  # > 1M tokens
            return Decimal('0.15')  # 15% discount
        elif total_tokens > 500000:  # > 500K tokens
            return Decimal('0.10')  # 10% discount
        elif total_tokens > 100000:  # > 100K tokens
            return Decimal('0.05')  # 5% discount

        return Decimal('0')  # No discount

    def estimate_monthly_cost(self, model: LLMModel,
                            estimated_requests_per_day: int,
                            avg_input_tokens: int,
                            avg_output_tokens: int,
                            operation: OperationType) -> Dict[str, Any]:
        """Estima custo mensal baseado em uso estimado"""
        daily_cost = Decimal(str(model.calculate_operation_cost(
            operation, avg_input_tokens, avg_output_tokens
        ))) * estimated_requests_per_day

        monthly_cost = daily_cost * 30

        # Considera desconto por volume mensal
        monthly_tokens = (avg_input_tokens + avg_output_tokens) * estimated_requests_per_day * 30
        volume_discount = self._calculate_volume_discount(monthly_tokens)
        discounted_monthly_cost = monthly_cost * (1 - volume_discount)

        return {
            'estimated_monthly_cost': float(discounted_monthly_cost),
            'cost_without_discount': float(monthly_cost),
            'volume_discount_applied': float(volume_discount),
            'daily_cost': float(daily_cost),
            'cost_per_request': float(daily_cost / estimated_requests_per_day),
            'monthly_token_volume': monthly_tokens
        }

    def compare_model_costs(self, models: List[LLMModel],
                          operation: OperationType,
                          input_tokens: int,
                          output_tokens: int) -> List[Dict[str, Any]]:
        """Compara custos entre múltiplos modelos"""
        cost_comparisons = []

        for model in models:
            if operation not in model.supported_operations:
                continue

            try:
                cost = model.calculate_operation_cost(operation, input_tokens, output_tokens)

                cost_comparisons.append({
                    'model_id': model.id,
                    'model_name': model.name,
                    'cost': cost,
                    'provider': model.provider_config.provider_type.value,
                    'performance_tier': model.metadata.performance_tier.value,
                    'quality_score': model.metadata.quality_score,
                    'cost_per_quality_point': cost / model.metadata.quality_score if model.metadata.quality_score else None
                })
            except Exception:
                continue  # Skip models that can't calculate cost

        # Ordena por custo
        cost_comparisons.sort(key=lambda x: x['cost'])

        # Adiciona ranking
        for i, comparison in enumerate(cost_comparisons):
            comparison['cost_rank'] = i + 1

        return cost_comparisons

    def calculate_cost_efficiency_score(self, model: LLMModel,
                                      operation: OperationType) -> float:
        """Calcula score de eficiência de custo (qualidade/preço)"""
        if not model.metadata.quality_score:
            return 0.0

        # Usa tokens médios para cálculo base
        sample_cost = model.calculate_operation_cost(operation, 1000, 500)

        if sample_cost == 0:
            return float('inf')  # Gratuito = eficiência infinita

        # Score = qualidade / custo (normalizado)
        efficiency_score = model.metadata.quality_score / (sample_cost * 1000)

        return efficiency_score
```

## 📚 Interfaces de Repositório

### domain/repositories/llm_model_repository.py
```python
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from domain.entities.llm_model import LLMModel
from domain.value_objects.provider_config import ProviderType

class LLMModelRepositoryInterface(ABC):
    """Interface para repositório de modelos LLM"""

    @abstractmethod
    async def save(self, model: LLMModel) -> LLMModel:
        """Salva um modelo (create ou update)"""
        pass

    @abstractmethod
    async def find_by_id(self, model_id: str) -> Optional[LLMModel]:
        """Busca modelo por ID"""
        pass

    @abstractmethod
    async def find_all_active(self) -> List[LLMModel]:
        """Busca todos os modelos ativos"""
        pass

    @abstractmethod
    async def find_by_provider(self, provider_type: ProviderType) -> List[LLMModel]:
        """Busca modelos por provedor"""
        pass

    @abstractmethod
    async def find_by_project_access(self, project_id: str) -> List[LLMModel]:
        """Busca modelos acessíveis a um projeto"""
        pass

    @abstractmethod
    async def find_supporting_operation(self, operation_name: str) -> List[LLMModel]:
        """Busca modelos que suportam uma operação específica"""
        pass

    @abstractmethod
    async def delete(self, model_id: str) -> bool:
        """Remove um modelo"""
        pass

    @abstractmethod
    async def exists(self, model_id: str) -> bool:
        """Verifica se um modelo existe"""
        pass
```

## 🎭 Eventos de Domínio

### domain/events/model_events.py
```python
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

@dataclass(frozen=True)
class DomainEvent:
    """Evento base do domínio"""
    event_id: str
    occurred_at: datetime
    aggregate_id: str

@dataclass(frozen=True)
class ModelCreated(DomainEvent):
    """Evento: modelo foi criado"""
    model_name: str
    provider_type: str
    created_by: Optional[str] = None

@dataclass(frozen=True)
class ModelUpdated(DomainEvent):
    """Evento: modelo foi atualizado"""
    updated_fields: list
    updated_by: str
    old_values: Dict[str, Any]
    new_values: Dict[str, Any]

@dataclass(frozen=True)
class ModelStatusChanged(DomainEvent):
    """Evento: status do modelo mudou"""
    old_status: str
    new_status: str
    changed_by: str
    reason: Optional[str] = None

@dataclass(frozen=True)
class ModelOperationAdded(DomainEvent):
    """Evento: operação foi adicionada ao modelo"""
    operation_name: str
    added_by: str

@dataclass(frozen=True)
class ModelOperationRemoved(DomainEvent):
    """Evento: operação foi removida do modelo"""
    operation_name: str
    removed_by: str
    reason: Optional[str] = None
```

## 🎯 Resumo das Correções DDD

### ✅ **Melhorias Implementadas:**

1. **Value Objects Ricos** - Validação e comportamentos específicos
2. **Entidades com Comportamentos** - Lógica de negócio rica
3. **Serviços de Domínio Especializados** - Lógicas complexas separadas
4. **Linguagem Ubíqua Consistente** - Termos do domínio bem definidos
5. **Agregados Bem Delimitados** - Invariantes e consistência
6. **Eventos de Domínio** - Comunicação entre bounded contexts
7. **Interfaces de Repositório** - Inversão de dependência correta

### 🚀 **Benefícios da Correção:**

- **Manutenibilidade** - Código mais expressivo e fácil de evoluir
- **Testabilidade** - Unidades bem isoladas e testáveis
- **Flexibilidade** - Mudanças de regras isoladas no domínio
- **Qualidade** - Validações ricas e comportamentos corretos
- **Performance** - Cálculos otimizados nos serviços especializados

A implementação agora segue corretamente os princípios do DDD com uma separação clara entre camadas e responsabilidades bem definidas.