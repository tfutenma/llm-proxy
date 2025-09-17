# Arquitetura Hexagonal - Exemplo Prático Completo

## 🎯 Domínio de Exemplo: Sistema de Gerenciamento de Modelos LLM

Vamos implementar um sistema de gerenciamento de modelos LLM para demonstrar todos os conceitos da arquitetura hexagonal aplicados ao contexto de um proxy LLM. Este exemplo mostra como gerenciar modelos de diferentes provedores de IA com suas configurações, custos e operações suportadas.

### 📋 Requisitos do Sistema

**Funcionalidades Core:**
- Gerenciar modelos LLM (criar, buscar, atualizar, deletar)
- Listar modelos por provedor (Azure, OpenAI, Anthropic, etc.)
- Listar modelos por tipo (AzureOpenAI, OpenAI, AnthropicVertex, etc.)
- Listar modelos por operação suportada (ChatCompletion, Embeddings, etc.)
- Buscar modelo por nome, ID ou deployment_name
- Controlar acesso por projetos e privacidade
- Gerenciar custos e parâmetros de configuração

**Regras de Negócio:**
- Cada modelo deve ter um ID único
- Modelos podem ser públicos ou privados
- Modelos privados são restritos a projetos específicos
- Cada modelo possui configurações específicas do provedor
- Custos são calculados por token de entrada e saída
- Operações suportadas variam por modelo e provedor

## 🏗️ Racional da Arquitetura Hexagonal

### Por que Arquitetura Hexagonal?

1. **Independência de Framework**: O domínio não depende de Flask, FastAPI, Django, etc.
2. **Testabilidade**: Podemos testar a lógica de negócio sem banco de dados ou APIs
3. **Flexibilidade**: Podemos trocar banco de dados, APIs, ou interfaces sem afetar o core
4. **Manutenibilidade**: Código organizado em camadas com responsabilidades claras

### Estrutura das Camadas

```
📁 llm_model_management/
├── 📁 domain/              # CORE - Lógica de Negócio
│   ├── entities/           # Entidades do domínio
│   ├── value_objects/      # Objetos de valor
│   ├── services/           # Serviços de domínio
│   └── repositories/       # Interfaces dos repositórios
├── 📁 application/         # CASOS DE USO
│   ├── use_cases/          # Casos de uso da aplicação
│   ├── ports/              # Interfaces (Ports)
│   └── services/           # Serviços de aplicação
├── 📁 infrastructure/      # ADAPTADORES
│   ├── persistence/        # Adaptadores de banco de dados (CosmosDB)
│   ├── web/                # Adaptadores HTTP (FastAPI)
│   ├── cache/              # Adaptadores de cache
│   └── config/             # Configurações
└── 📁 tests/               # Testes automatizados
    ├── unit/               # Testes unitários
    ├── integration/        # Testes de integração
    └── e2e/                # Testes end-to-end
```

## 🔧 Implementação Completa

### 1. Camada de Domínio (Core)

#### `domain/entities/llm_model.py`
```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from ..value_objects.model_parameters import ModelParameters
from ..value_objects.cost_info import CostInfo
from ..value_objects.model_status import ModelStatus

@dataclass
class LLMModel:
    """Entidade central do domínio: Modelo LLM"""

    id: str
    name: str
    code: str
    provider: str
    model_type: str
    parameters: ModelParameters
    costs: CostInfo
    operations: List[str] = field(default_factory=list)
    projects: List[str] = field(default_factory=list)
    private: bool = False
    active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            raise ValueError("ID é obrigatório")
        if not self.name:
            raise ValueError("Nome é obrigatório")
        if not self.code:
            raise ValueError("Código é obrigatório")
        if not self.provider:
            raise ValueError("Provedor é obrigatório")
        if not self.model_type:
            raise ValueError("Tipo do modelo é obrigatório")
        if not self.operations:
            raise ValueError("Pelo menos uma operação deve ser suportada")

    def is_accessible_by_project(self, project_id: str) -> bool:
        """Verifica se o modelo é acessível pelo projeto"""
        if not self.private:
            return True
        return project_id in self.projects

    def supports_operation(self, operation: str) -> bool:
        """Verifica se o modelo suporta a operação"""
        return operation in self.operations

    def add_project(self, project_id: str) -> None:
        """Adiciona projeto ao modelo"""
        if project_id not in self.projects:
            self.projects.append(project_id)
            self.updated_at = datetime.utcnow()

    def remove_project(self, project_id: str) -> None:
        """Remove projeto do modelo"""
        if project_id in self.projects:
            self.projects.remove(project_id)
            self.updated_at = datetime.utcnow()

    def update_parameters(self, new_parameters: Dict[str, Any]) -> None:
        """Atualiza parâmetros do modelo"""
        self.parameters = ModelParameters.from_dict(new_parameters)
        self.updated_at = datetime.utcnow()

    def update_costs(self, input_cost: float, output_cost: float, currency: str = "USD") -> None:
        """Atualiza custos do modelo"""
        self.costs = CostInfo(currency, input_cost, output_cost)
        self.updated_at = datetime.utcnow()

    def deactivate(self) -> None:
        """Desativa o modelo"""
        self.active = False
        self.updated_at = datetime.utcnow()

    def activate(self) -> None:
        """Ativa o modelo"""
        self.active = True
        self.updated_at = datetime.utcnow()

    def get_status(self) -> ModelStatus:
        """Retorna o status atual do modelo"""
        if not self.active:
            return ModelStatus.INACTIVE
        elif self.private and not self.projects:
            return ModelStatus.PRIVATE_NO_PROJECTS
        elif self.private:
            return ModelStatus.PRIVATE
        else:
            return ModelStatus.PUBLIC
```

#### `domain/value_objects/model_parameters.py`
```python
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass(frozen=True)
class ModelParameters:
    """Value Object: Parâmetros de configuração do modelo"""

    secret_name: str
    deployment_name: Optional[str] = None
    api_version: Optional[str] = None
    endpoint: Optional[str] = None
    enable_tools: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    additional_params: Dict[str, Any] = None

    def __post_init__(self):
        if not self.secret_name:
            raise ValueError("Secret name é obrigatório")

        if self.additional_params is None:
            object.__setattr__(self, 'additional_params', {})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelParameters":
        """Cria ModelParameters a partir de dicionário"""
        known_fields = {
            'secret_name', 'deployment_name', 'api_version',
            'endpoint', 'enable_tools', 'max_tokens', 'temperature'
        }

        # Separa campos conhecidos dos adicionais
        known_params = {k: v for k, v in data.items() if k in known_fields}
        additional_params = {k: v for k, v in data.items() if k not in known_fields}

        return cls(
            **known_params,
            additional_params=additional_params
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        result = {
            "secret_name": self.secret_name,
            "enable_tools": self.enable_tools
        }

        if self.deployment_name:
            result["deployment_name"] = self.deployment_name
        if self.api_version:
            result["api_version"] = self.api_version
        if self.endpoint:
            result["endpoint"] = self.endpoint
        if self.max_tokens:
            result["max_tokens"] = self.max_tokens
        if self.temperature:
            result["temperature"] = self.temperature

        result.update(self.additional_params)
        return result

    def get_deployment_name(self) -> Optional[str]:
        """Retorna deployment_name se disponível"""
        return self.deployment_name
```

#### `domain/value_objects/cost_info.py`
```python
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class CostInfo:
    """Value Object: Informações de custo do modelo"""

    currency: str
    cost_input_1Mtokens: float
    cost_output_1Mtokens: float

    def __post_init__(self):
        if not self.currency:
            raise ValueError("Moeda é obrigatória")
        if self.cost_input_1Mtokens < 0:
            raise ValueError("Custo de entrada não pode ser negativo")
        if self.cost_output_1Mtokens < 0:
            raise ValueError("Custo de saída não pode ser negativo")

        # Validação de moeda
        valid_currencies = {"USD", "EUR", "BRL", "GBP"}
        if self.currency not in valid_currencies:
            raise ValueError(f"Moeda inválida: {self.currency}. Suportadas: {valid_currencies}")

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calcula custo total baseado nos tokens"""
        input_cost = (input_tokens / 1_000_000) * self.cost_input_1Mtokens
        output_cost = (output_tokens / 1_000_000) * self.cost_output_1Mtokens
        return input_cost + output_cost

    def get_input_cost_per_token(self) -> float:
        """Retorna custo por token de entrada"""
        return self.cost_input_1Mtokens / 1_000_000

    def get_output_cost_per_token(self) -> float:
        """Retorna custo por token de saída"""
        return self.cost_output_1Mtokens / 1_000_000

    def to_dict(self) -> dict:
        """Converte para dicionário"""
        return {
            "currency": self.currency,
            "cost_input_1Mtokens": self.cost_input_1Mtokens,
            "cost_output_1Mtokens": self.cost_output_1Mtokens
        }
```

#### `domain/value_objects/enums.py`
```python
from enum import Enum

class ModelStatus(Enum):
    """Status do modelo LLM"""
    PUBLIC = "public"
    PRIVATE = "private"
    PRIVATE_NO_PROJECTS = "private_no_projects"
    INACTIVE = "inactive"

class ProviderType(Enum):
    """Tipos de provedor"""
    AZURE = "Azure"
    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"
    GOOGLE = "Google"
    AWS = "AWS"
    HUGGING_FACE = "HuggingFace"
    COHERE = "Cohere"
    TOGETHER = "Together"

class ModelType(Enum):
    """Tipos de modelo"""
    AZURE_OPENAI = "AzureOpenAI"
    OPENAI = "OpenAI"
    ANTHROPIC_VERTEX = "AnthropicVertex"
    ANTHROPIC = "Anthropic"
    VERTEX_AI = "VertexAI"
    BEDROCK = "Bedrock"
    HUGGING_FACE = "HuggingFace"
    COHERE = "Cohere"
    TOGETHER_AI = "TogetherAI"

class OperationType(Enum):
    """Tipos de operação suportadas"""
    CHAT_COMPLETION = "ChatCompletion"
    COMPLETION = "Completion"
    EMBEDDINGS = "Embeddings"
    IMAGE_GENERATION = "ImageGeneration"
    SPEECH_TO_TEXT = "SpeechToText"
    TEXT_TO_SPEECH = "TextToSpeech"
    MODERATION = "Moderation"
    RESPONSES = "Responses"
```

#### `domain/services/model_validation_service.py`
```python
from typing import List, Dict, Any
from ..entities.llm_model import LLMModel
from ..value_objects.enums import ProviderType, ModelType, OperationType

class ModelValidationService:
    """Serviço de domínio para validação de modelos"""

    PROVIDER_MODEL_TYPE_MAPPING = {
        ProviderType.AZURE.value: [ModelType.AZURE_OPENAI.value],
        ProviderType.OPENAI.value: [ModelType.OPENAI.value],
        ProviderType.ANTHROPIC.value: [ModelType.ANTHROPIC.value, ModelType.ANTHROPIC_VERTEX.value],
        ProviderType.GOOGLE.value: [ModelType.VERTEX_AI.value],
        ProviderType.AWS.value: [ModelType.BEDROCK.value],
        ProviderType.HUGGING_FACE.value: [ModelType.HUGGING_FACE.value],
        ProviderType.COHERE.value: [ModelType.COHERE.value],
        ProviderType.TOGETHER.value: [ModelType.TOGETHER_AI.value]
    }

    REQUIRED_PARAMETERS_BY_TYPE = {
        ModelType.AZURE_OPENAI.value: ["deployment_name", "api_version", "endpoint"],
        ModelType.OPENAI.value: [],
        ModelType.ANTHROPIC.value: [],
        ModelType.VERTEX_AI.value: ["project_id", "location"],
        ModelType.BEDROCK.value: ["region"]
    }

    @classmethod
    def validate_provider_model_type_compatibility(cls, provider: str, model_type: str) -> bool:
        """Valida se o tipo de modelo é compatível com o provedor"""
        allowed_types = cls.PROVIDER_MODEL_TYPE_MAPPING.get(provider, [])
        return model_type in allowed_types

    @classmethod
    def validate_required_parameters(cls, model_type: str, parameters: Dict[str, Any]) -> List[str]:
        """Valida parâmetros obrigatórios e retorna lista de erros"""
        required_params = cls.REQUIRED_PARAMETERS_BY_TYPE.get(model_type, [])
        missing_params = []

        for param in required_params:
            if param not in parameters or not parameters[param]:
                missing_params.append(param)

        return missing_params

    @classmethod
    def validate_operations(cls, operations: List[str]) -> List[str]:
        """Valida operações suportadas"""
        valid_operations = [op.value for op in OperationType]
        invalid_operations = [op for op in operations if op not in valid_operations]
        return invalid_operations

    @classmethod
    def validate_model(cls, model: LLMModel) -> List[str]:
        """Validação completa do modelo"""
        errors = []

        # Valida compatibilidade provedor-tipo
        if not cls.validate_provider_model_type_compatibility(model.provider, model.model_type):
            errors.append(f"Tipo de modelo '{model.model_type}' não é compatível com provedor '{model.provider}'")

        # Valida parâmetros obrigatórios
        missing_params = cls.validate_required_parameters(model.model_type, model.parameters.to_dict())
        if missing_params:
            errors.append(f"Parâmetros obrigatórios ausentes: {', '.join(missing_params)}")

        # Valida operações
        invalid_ops = cls.validate_operations(model.operations)
        if invalid_ops:
            errors.append(f"Operações inválidas: {', '.join(invalid_ops)}")

        return errors
```

#### `domain/repositories/model_repository.py`
```python
from abc import ABC, abstractmethod
from typing import List, Optional
from ..entities.llm_model import LLMModel

class ModelRepositoryPort(ABC):
    """Port para repositório de modelos LLM"""

    @abstractmethod
    async def save(self, model: LLMModel) -> LLMModel:
        """Salva ou atualiza um modelo"""
        pass

    @abstractmethod
    async def find_by_id(self, model_id: str) -> Optional[LLMModel]:
        """Busca modelo por ID"""
        pass

    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[LLMModel]:
        """Busca modelo por nome"""
        pass

    @abstractmethod
    async def find_by_deployment_name(self, deployment_name: str) -> Optional[LLMModel]:
        """Busca modelo por deployment_name"""
        pass

    @abstractmethod
    async def find_by_provider(self, provider: str) -> List[LLMModel]:
        """Lista modelos por provedor"""
        pass

    @abstractmethod
    async def find_by_model_type(self, model_type: str) -> List[LLMModel]:
        """Lista modelos por tipo"""
        pass

    @abstractmethod
    async def find_by_operation(self, operation: str) -> List[LLMModel]:
        """Lista modelos que suportam uma operação"""
        pass

    @abstractmethod
    async def find_by_project(self, project_id: str) -> List[LLMModel]:
        """Lista modelos acessíveis por um projeto"""
        pass

    @abstractmethod
    async def find_all(self, include_inactive: bool = False) -> List[LLMModel]:
        """Lista todos os modelos"""
        pass

    @abstractmethod
    async def delete(self, model_id: str) -> bool:
        """Remove um modelo"""
        pass

    @abstractmethod
    async def exists_by_id(self, model_id: str) -> bool:
        """Verifica se modelo existe por ID"""
        pass
```

### 2. Camada de Aplicação (Use Cases)

#### `application/ports/repositories.py`
```python
from abc import ABC, abstractmethod
from typing import List, Optional

from domain.entities.llm_model import LLMModel
from domain.repositories.model_repository import ModelRepositoryPort

# Re-exporta a interface do domínio para a camada de aplicação
class ModelRepositoryPort(ModelRepositoryPort):
    """Port para repositório de modelos na camada de aplicação"""
    pass
```

#### `application/use_cases/model_management.py`
```python
from typing import List, Optional, Dict, Any
from datetime import datetime

from domain.entities.llm_model import LLMModel
from domain.value_objects.model_parameters import ModelParameters
from domain.value_objects.cost_info import CostInfo
from domain.services.model_validation_service import ModelValidationService
from ..ports.repositories import ModelRepositoryPort

class CreateModelUseCase:
    """Caso de uso: Criar modelo LLM"""

    def __init__(self, model_repository: ModelRepositoryPort):
        self._model_repository = model_repository
        self._validation_service = ModelValidationService()

    async def execute(
        self,
        id: str,
        name: str,
        code: str,
        provider: str,
        model_type: str,
        parameters: Dict[str, Any],
        costs: Dict[str, Any],
        operations: List[str],
        projects: List[str] = None,
        private: bool = False
    ) -> LLMModel:
        """Executa a criação de um modelo"""

        # Verifica se já existe modelo com o mesmo ID
        if await self._model_repository.exists_by_id(id):
            raise ValueError(f"Já existe um modelo com ID {id}")

        # Cria objetos de valor
        model_params = ModelParameters.from_dict(parameters)
        cost_info = CostInfo(
            currency=costs.get("currency", "USD"),
            cost_input_1Mtokens=costs["cost_input_1Mtokens"],
            cost_output_1Mtokens=costs["cost_output_1Mtokens"]
        )

        # Cria o modelo
        model = LLMModel(
            id=id,
            name=name,
            code=code,
            provider=provider,
            model_type=model_type,
            parameters=model_params,
            costs=cost_info,
            operations=operations,
            projects=projects or [],
            private=private
        )

        # Valida o modelo
        validation_errors = self._validation_service.validate_model(model)
        if validation_errors:
            raise ValueError(f"Erros de validação: {'; '.join(validation_errors)}")

        # Salva no repositório
        return await self._model_repository.save(model)

class FindModelUseCase:
    """Caso de uso: Buscar modelos"""

    def __init__(self, model_repository: ModelRepositoryPort):
        self._model_repository = model_repository

    async def execute_by_id(self, model_id: str) -> Optional[LLMModel]:
        """Busca modelo por ID"""
        return await self._model_repository.find_by_id(model_id)

    async def execute_by_name(self, name: str) -> Optional[LLMModel]:
        """Busca modelo por nome"""
        return await self._model_repository.find_by_name(name)

    async def execute_by_deployment_name(self, deployment_name: str) -> Optional[LLMModel]:
        """Busca modelo por deployment_name"""
        return await self._model_repository.find_by_deployment_name(deployment_name)

    async def execute_by_provider(self, provider: str) -> List[LLMModel]:
        """Lista modelos por provedor"""
        return await self._model_repository.find_by_provider(provider)

    async def execute_by_model_type(self, model_type: str) -> List[LLMModel]:
        """Lista modelos por tipo"""
        return await self._model_repository.find_by_model_type(model_type)

    async def execute_by_operation(self, operation: str) -> List[LLMModel]:
        """Lista modelos que suportam uma operação"""
        return await self._model_repository.find_by_operation(operation)

    async def execute_by_project(self, project_id: str) -> List[LLMModel]:
        """Lista modelos acessíveis por projeto"""
        return await self._model_repository.find_by_project(project_id)

    async def execute_all(self, include_inactive: bool = False) -> List[LLMModel]:
        """Lista todos os modelos"""
        return await self._model_repository.find_all(include_inactive)

class UpdateModelUseCase:
    """Caso de uso: Atualizar modelo"""

    def __init__(self, model_repository: ModelRepositoryPort):
        self._model_repository = model_repository
        self._validation_service = ModelValidationService()

    async def execute(
        self,
        model_id: str,
        name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        costs: Optional[Dict[str, Any]] = None,
        operations: Optional[List[str]] = None,
        projects: Optional[List[str]] = None,
        private: Optional[bool] = None,
        active: Optional[bool] = None
    ) -> LLMModel:
        """Atualiza um modelo existente"""

        model = await self._model_repository.find_by_id(model_id)
        if not model:
            raise ValueError(f"Modelo com ID {model_id} não encontrado")

        # Atualiza campos se fornecidos
        if name is not None:
            model.name = name
        if parameters is not None:
            model.update_parameters(parameters)
        if costs is not None:
            model.update_costs(
                input_cost=costs["cost_input_1Mtokens"],
                output_cost=costs["cost_output_1Mtokens"],
                currency=costs.get("currency", model.costs.currency)
            )
        if operations is not None:
            model.operations = operations
        if projects is not None:
            model.projects = projects
        if private is not None:
            model.private = private
        if active is not None:
            if active:
                model.activate()
            else:
                model.deactivate()

        # Valida o modelo atualizado
        validation_errors = self._validation_service.validate_model(model)
        if validation_errors:
            raise ValueError(f"Erros de validação: {'; '.join(validation_errors)}")

        return await self._model_repository.save(model)

class DeleteModelUseCase:
    """Caso de uso: Deletar modelo"""

    def __init__(self, model_repository: ModelRepositoryPort):
        self._model_repository = model_repository

    async def execute(self, model_id: str) -> bool:
        """Deleta um modelo"""
        if not await self._model_repository.exists_by_id(model_id):
            raise ValueError(f"Modelo com ID {model_id} não encontrado")

        return await self._model_repository.delete(model_id)
```

### 3. Camada de Infraestrutura (Adapters)

#### `infrastructure/persistence/cosmosdb_model_repository.py`
```python
from typing import List, Optional, Dict, Any
from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosResourceNotFoundError

from domain.entities.llm_model import LLMModel
from domain.value_objects.model_parameters import ModelParameters
from domain.value_objects.cost_info import CostInfo
from application.ports.repositories import ModelRepositoryPort

class CosmosDBModelRepository(ModelRepositoryPort):
    """Implementação do repositório de modelos com CosmosDB"""

    def __init__(self, cosmos_client: CosmosClient, database_name: str, container_name: str):
        self._client = cosmos_client
        self._database = cosmos_client.get_database_client(database_name)
        self._container = self._database.get_container_client(container_name)

    async def save(self, model: LLMModel) -> LLMModel:
        """Salva um modelo no CosmosDB"""
        document = self._to_document(model)

        try:
            # Tenta atualizar se existe
            self._container.upsert_item(document)
        except Exception as e:
            raise RuntimeError(f"Erro ao salvar modelo: {str(e)}")

        return model

    async def find_by_id(self, model_id: str) -> Optional[LLMModel]:
        """Busca modelo por ID"""
        try:
            item = self._container.read_item(item=model_id, partition_key=model_id)
            return self._to_entity(item)
        except CosmosResourceNotFoundError:
            return None

    async def find_by_name(self, name: str) -> Optional[LLMModel]:
        """Busca modelo por nome"""
        query = "SELECT * FROM c WHERE c.name = @name"
        parameters = [{"name": "@name", "value": name}]

        items = list(self._container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))

        return self._to_entity(items[0]) if items else None

    async def find_by_deployment_name(self, deployment_name: str) -> Optional[LLMModel]:
        """Busca modelo por deployment_name"""
        query = "SELECT * FROM c WHERE c.parameters.deployment_name = @deployment_name"
        parameters = [{"name": "@deployment_name", "value": deployment_name}]

        items = list(self._container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))

        return self._to_entity(items[0]) if items else None

    async def find_by_provider(self, provider: str) -> List[LLMModel]:
        """Lista modelos por provedor"""
        query = "SELECT * FROM c WHERE c.provider = @provider AND c.active = true"
        parameters = [{"name": "@provider", "value": provider}]

        items = list(self._container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))

        return [self._to_entity(item) for item in items]

    async def find_by_model_type(self, model_type: str) -> List[LLMModel]:
        """Lista modelos por tipo"""
        query = "SELECT * FROM c WHERE c.model_type = @model_type AND c.active = true"
        parameters = [{"name": "@model_type", "value": model_type}]

        items = list(self._container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))

        return [self._to_entity(item) for item in items]

    async def find_by_operation(self, operation: str) -> List[LLMModel]:
        """Lista modelos que suportam uma operação"""
        query = "SELECT * FROM c WHERE ARRAY_CONTAINS(c.operations, @operation) AND c.active = true"
        parameters = [{"name": "@operation", "value": operation}]

        items = list(self._container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))

        return [self._to_entity(item) for item in items]

    async def find_by_project(self, project_id: str) -> List[LLMModel]:
        """Lista modelos acessíveis por um projeto"""
        query = """SELECT * FROM c WHERE c.active = true AND
                    (c.private = false OR ARRAY_CONTAINS(c.projects, @project_id))"""
        parameters = [{"name": "@project_id", "value": project_id}]

        items = list(self._container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))

        return [self._to_entity(item) for item in items]

    async def find_all(self, include_inactive: bool = False) -> List[LLMModel]:
        """Lista todos os modelos"""
        if include_inactive:
            query = "SELECT * FROM c"
            parameters = []
        else:
            query = "SELECT * FROM c WHERE c.active = true"
            parameters = []

        items = list(self._container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))

        return [self._to_entity(item) for item in items]

    async def delete(self, model_id: str) -> bool:
        """Remove um modelo"""
        try:
            self._container.delete_item(item=model_id, partition_key=model_id)
            return True
        except CosmosResourceNotFoundError:
            return False

    async def exists_by_id(self, model_id: str) -> bool:
        """Verifica se modelo existe por ID"""
        try:
            self._container.read_item(item=model_id, partition_key=model_id)
            return True
        except CosmosResourceNotFoundError:
            return False

    def _to_document(self, model: LLMModel) -> Dict[str, Any]:
        """Converte entidade para documento CosmosDB"""
        return {
            "id": model.id,
            "name": model.name,
            "code": model.code,
            "provider": model.provider,
            "model_type": model.model_type,
            "parameters": model.parameters.to_dict(),
            "costs": model.costs.to_dict(),
            "operations": model.operations,
            "projects": model.projects,
            "private": model.private,
            "active": model.active,
            "created_at": model.created_at.isoformat(),
            "updated_at": model.updated_at.isoformat()
        }

    def _to_entity(self, document: Dict[str, Any]) -> LLMModel:
        """Converte documento para entidade"""
        from datetime import datetime

        parameters = ModelParameters.from_dict(document["parameters"])
        costs = CostInfo(**document["costs"])

        return LLMModel(
            id=document["id"],
            name=document["name"],
            code=document["code"],
            provider=document["provider"],
            model_type=document["model_type"],
            parameters=parameters,
            costs=costs,
            operations=document["operations"],
            projects=document["projects"],
            private=document["private"],
            active=document["active"],
            created_at=datetime.fromisoformat(document["created_at"]),
            updated_at=datetime.fromisoformat(document["updated_at"])
        )
```

#### `infrastructure/web/fastapi_controllers.py`
```python
from fastapi import APIRouter, HTTPException, Depends, status, Query
from typing import List, Optional

from application.use_cases.model_management import (
    CreateModelUseCase, FindModelUseCase, UpdateModelUseCase, DeleteModelUseCase
)
from .schemas import (
    ModelCreateRequest, ModelResponse, ModelUpdateRequest
)
from .dependencies import get_model_use_cases

router = APIRouter()

# Endpoints de Modelos
@router.post("/models", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def create_model(
    request: ModelCreateRequest,
    use_cases = Depends(get_model_use_cases)
):
    """Cria um novo modelo LLM"""
    try:
        create_use_case: CreateModelUseCase = use_cases["create"]
        model = await create_use_case.execute(
            id=request.id,
            name=request.name,
            code=request.code,
            provider=request.provider,
            model_type=request.model_type,
            parameters=request.parameters,
            costs=request.costs,
            operations=request.operations,
            projects=request.projects,
            private=request.private
        )
        return ModelResponse.from_entity(model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/models/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: str,
    use_cases = Depends(get_model_use_cases)
):
    """Busca modelo por ID"""
    find_use_case: FindModelUseCase = use_cases["find"]
    model = await find_use_case.execute_by_id(model_id)

    if not model:
        raise HTTPException(status_code=404, detail="Modelo não encontrado")

    return ModelResponse.from_entity(model)

@router.get("/models", response_model=List[ModelResponse])
async def list_models(
    provider: Optional[str] = Query(None, description="Filtrar por provedor"),
    model_type: Optional[str] = Query(None, description="Filtrar por tipo de modelo"),
    operation: Optional[str] = Query(None, description="Filtrar por operação suportada"),
    project_id: Optional[str] = Query(None, description="Filtrar por projeto"),
    include_inactive: bool = Query(False, description="Incluir modelos inativos"),
    use_cases = Depends(get_model_use_cases)
):
    """Lista modelos com filtros opcionais"""
    find_use_case: FindModelUseCase = use_cases["find"]

    if provider:
        models = await find_use_case.execute_by_provider(provider)
    elif model_type:
        models = await find_use_case.execute_by_model_type(model_type)
    elif operation:
        models = await find_use_case.execute_by_operation(operation)
    elif project_id:
        models = await find_use_case.execute_by_project(project_id)
    else:
        models = await find_use_case.execute_all(include_inactive)

    return [ModelResponse.from_entity(model) for model in models]

@router.get("/models/by-name/{name}", response_model=ModelResponse)
async def get_model_by_name(
    name: str,
    use_cases = Depends(get_model_use_cases)
):
    """Busca modelo por nome"""
    find_use_case: FindModelUseCase = use_cases["find"]
    model = await find_use_case.execute_by_name(name)

    if not model:
        raise HTTPException(status_code=404, detail="Modelo não encontrado")

    return ModelResponse.from_entity(model)

@router.get("/models/by-deployment/{deployment_name}", response_model=ModelResponse)
async def get_model_by_deployment(
    deployment_name: str,
    use_cases = Depends(get_model_use_cases)
):
    """Busca modelo por deployment_name"""
    find_use_case: FindModelUseCase = use_cases["find"]
    model = await find_use_case.execute_by_deployment_name(deployment_name)

    if not model:
        raise HTTPException(status_code=404, detail="Modelo não encontrado")

    return ModelResponse.from_entity(model)

@router.patch("/models/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: str,
    request: ModelUpdateRequest,
    use_cases = Depends(get_model_use_cases)
):
    """Atualiza um modelo existente"""
    try:
        update_use_case: UpdateModelUseCase = use_cases["update"]
        model = await update_use_case.execute(
            model_id=model_id,
            name=request.name,
            parameters=request.parameters,
            costs=request.costs,
            operations=request.operations,
            projects=request.projects,
            private=request.private,
            active=request.active
        )
        return ModelResponse.from_entity(model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: str,
    use_cases = Depends(get_model_use_cases)
):
    """Deleta um modelo"""
    try:
        delete_use_case: DeleteModelUseCase = use_cases["delete"]
        success = await delete_use_case.execute(model_id)
        if not success:
            raise HTTPException(status_code=404, detail="Modelo não encontrado")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

#### `infrastructure/web/schemas.py`
```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any

from domain.entities.llm_model import LLMModel

class ModelCreateRequest(BaseModel):
    """Schema para criação de modelo"""
    id: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=255)
    code: str = Field(..., min_length=1, max_length=100)
    provider: str = Field(..., min_length=1, max_length=50)
    model_type: str = Field(..., min_length=1, max_length=50)
    parameters: Dict[str, Any] = Field(...)
    costs: Dict[str, Any] = Field(...)
    operations: List[str] = Field(..., min_items=1)
    projects: List[str] = Field(default_factory=list)
    private: bool = Field(default=False)

class ModelUpdateRequest(BaseModel):
    """Schema para atualização de modelo"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    parameters: Optional[Dict[str, Any]] = None
    costs: Optional[Dict[str, Any]] = None
    operations: Optional[List[str]] = None
    projects: Optional[List[str]] = None
    private: Optional[bool] = None
    active: Optional[bool] = None

class ModelResponse(BaseModel):
    """Schema de resposta para modelo"""
    id: str
    name: str
    code: str
    provider: str
    model_type: str
    parameters: Dict[str, Any]
    costs: Dict[str, Any]
    operations: List[str]
    projects: List[str]
    private: bool
    active: bool
    status: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_entity(cls, model: LLMModel) -> "ModelResponse":
        return cls(
            id=model.id,
            name=model.name,
            code=model.code,
            provider=model.provider,
            model_type=model.model_type,
            parameters=model.parameters.to_dict(),
            costs=model.costs.to_dict(),
            operations=model.operations,
            projects=model.projects,
            private=model.private,
            active=model.active,
            status=model.get_status().value,
            created_at=model.created_at,
            updated_at=model.updated_at
        )

class ModelListResponse(BaseModel):
    """Schema de resposta para lista de modelos"""
    models: List[ModelResponse]
    total: int
    provider_filter: Optional[str] = None
    model_type_filter: Optional[str] = None
    operation_filter: Optional[str] = None

class ProviderSummary(BaseModel):
    """Sumário por provedor"""
    provider: str
    model_count: int
    model_types: List[str]
    operations: List[str]

class OperationSummary(BaseModel):
    """Sumário por operação"""
    operation: str
    model_count: int
    providers: List[str]
    avg_input_cost: float
    avg_output_cost: float
```

### 4. Configuração e Dependency Injection

#### `infrastructure/config/dependencies.py`
```python
from functools import lru_cache
from azure.cosmos import CosmosClient
from typing import Dict, Any

from application.use_cases.model_management import (
    CreateModelUseCase, FindModelUseCase, UpdateModelUseCase, DeleteModelUseCase
)
from infrastructure.persistence.cosmosdb_model_repository import CosmosDBModelRepository
from .settings import get_settings

# Configuração do CosmosDB
@lru_cache()
def get_cosmos_client():
    settings = get_settings()
    return CosmosClient(
        url=settings.cosmos_endpoint,
        credential=settings.cosmos_key
    )

def get_model_repository():
    cosmos_client = get_cosmos_client()
    settings = get_settings()
    return CosmosDBModelRepository(
        cosmos_client=cosmos_client,
        database_name=settings.cosmos_database,
        container_name=settings.cosmos_container
    )

# Use Cases
def get_model_use_cases() -> Dict[str, Any]:
    model_repository = get_model_repository()
    return {
        "create": CreateModelUseCase(model_repository),
        "find": FindModelUseCase(model_repository),
        "update": UpdateModelUseCase(model_repository),
        "delete": DeleteModelUseCase(model_repository)
    }
```

#### `infrastructure/config/settings.py`
```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Configurações da aplicação"""

    # CosmosDB
    cosmos_endpoint: str
    cosmos_key: str
    cosmos_database: str = "llm_models"
    cosmos_container: str = "models"

    # Application
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    api_prefix: str = "/api/v1"

    # Logging
    log_level: str = "INFO"

    # Cache (Redis - opcional)
    redis_url: Optional[str] = None
    cache_ttl: int = 3600

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
```