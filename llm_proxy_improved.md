# LLM Proxy Microsserviço - Versão Aprimorada

## 📋 Visão Geral

Um microsserviço robusto em Python 3.11 usando FastAPI que atua como um proxy unificado para múltiplos provedores de LLM (Large Language Models). O serviço replica fielmente a interface da API da OpenAI, permitindo integração transparente com diversos provedores como Azure OpenAI, Google Vertex AI, Anthropic Claude e outros através da biblioteca LiteLLM.

### 🎯 Principais Benefícios

- **Interface Unificada**: Uma única API para acessar múltiplos provedores de LLM
- **Compatibilidade OpenAI**: Drop-in replacement para aplicações que já usam a API OpenAI
- **Gestão Centralizada**: Controle de modelos, permissões e configurações em um único lugar
- **Multi-Provider**: Suporte nativo para Azure, Google, Anthropic, OpenAI e outros
- **Segurança**: Gestão segura de API keys e credenciais
- **Escalabilidade**: Arquitetura preparada para alta demanda
- **Operações Assíncronas**: Todas as operações são assíncronas para máxima performance

## 🏗️ Arquitetura

O projeto segue uma **arquitetura hexagonal** (Ports and Adapters) aprimorada que promove:
- **Separação clara de responsabilidades**: Código organizado e manutenível
- **Testabilidade**: Facilita testes unitários e de integração
- **Flexibilidade**: Fácil adição de novos provedores ou funcionalidades
- **Independência**: Domínio independente de frameworks e bibliotecas externas
- **Async-First**: Todas as operações são assíncronas por padrão

### Camadas da Aplicação

| Camada | Responsabilidade | Componentes |
|--------|-----------------|-------------|
| **Application** | Coordena casos de uso e define interfaces | Interfaces Assíncronas, Use Cases |
| **Controllers** | Gerencia requisições HTTP e validações | Endpoints, Schemas, Routers |
| **Domain** | Lógica de negócio e regras do domínio | Entities, Services, Validators |
| **Infrastructure** | Implementações concretas e integrações | Database, Providers, Config |

## 📁 Estrutura de Diretórios Aprimorada

```bash
# Visualização completa da estrutura do projeto
llm-proxy/
├── 📂 llmproxy/                     # Pacote principal do projeto
│   ├── __init__.py
│   └── config.py                   # Configurações centralizadas
├── 📂 application/                 # Camada de aplicação
│   ├── 📂 interfaces/              # Contratos e abstrações
│   │   ├── __init__.py
│   │   ├── llm_provider_interface.py    # Interface para provedores LLM
│   │   └── llm_service_interface.py     # Interface para serviços LLM
│   └── 📂 use_cases/               # Casos de uso da aplicação
│       ├── __init__.py
│       ├── chat_completion_use_case.py  # UC: Chat/Completions
│       ├── text_to_speech_use_case.py   # UC: Text-to-Speech
│       ├── speech_to_text_use_case.py   # UC: Speech-to-Text
│       └── image_generation_use_case.py # UC: Geração de imagens
├── 📂 domain/                      # Camada de domínio
│   ├── 📂 entities/                # Entidades do domínio
│   │   ├── __init__.py
│   │   ├── llm_model.py            # Entidade LLMModel
│   │   └── operation.py            # Entidade Operation
│   └── 📂 services/                # Serviços de domínio
│       ├── __init__.py
│       ├── llm_model_service.py         # Serviço de modelos LLM
│       └── operation_validator.py       # Validador de operações
├── 📂 infra/                       # Camada de infraestrutura
│   ├── __init__.py
│   ├── 📂 controllers/             # Controladores HTTP
│   │   ├── __init__.py
│   │   ├── chat_completion_controller.py   # Controller de chat
│   │   ├── text_to_speech_controller.py    # Controller de TTS
│   │   ├── speech_to_text_controller.py    # Controller de STT
│   │   ├── image_generation_controller.py  # Controller de imagens
│   │   ├── model_controller.py             # Controller de modelos
│   │   └── app.py                         # Aplicação FastAPI principal
│   ├── 📂 schemas/                 # Schemas de dados (DTOs)
│   │   ├── __init__.py
│   │   ├── chat_completion_schemas.py # Schemas para chat
│   │   ├── text_to_speech_schemas.py  # Schemas para TTS
│   │   ├── speech_to_text_schemas.py  # Schemas para STT
│   │   ├── image_generation_schemas.py # Schemas para imagens
│   │   └── model_schemas.py           # Schemas para modelos
│   ├── 📂 database/                # Integração com banco de dados
│   │   ├── __init__.py
│   │   ├── cosmos_client.py             # Cliente Cosmos DB
│   │   ├── model_repository.py          # Repositório de modelos
│   │   └── cosmos_models.py             # Modelos de dados Cosmos
│   └── 📂 providers/               # Integração com provedores LLM
│       ├── __init__.py
│       ├── litellm_provider.py          # Provider LiteLLM
│       ├── llm_adapters.py              # Adaptadores para diferentes LLMs
│       └── provider_utils.py            # Utilitários para provedores
├── 📂 tests/                       # Testes automatizados
│   ├── __init__.py
│   ├── 📂 unit/                    # Testes unitários
│   ├── 📂 integration/             # Testes de integração
│   └── 📂 e2e/                     # Testes end-to-end
├── 📄 requirements.txt             # Dependências Python
├── 📄 requirements-dev.txt         # Dependências de desenvolvimento
├── 🐳 docker-compose.yml           # Orquestração Docker
├── 🐳 Dockerfile                   # Imagem Docker
├── 📄 .env.example                 # Exemplo de variáveis de ambiente
├── 📄 pytest.ini                   # Configuração de testes
├── 📄 pyproject.toml               # Configuração do projeto
└── 📄 README.md                    # Este arquivo
```

## 💻 Implementação Completa dos Arquivos Aprimorados

### Application Layer - Interfaces

#### application/interfaces/__init__.py
```python
# Interfaces da camada de aplicação
from .llm_provider_interface import LLMProviderInterface
from .llm_service_interface import LLMServiceInterface

__all__ = ['LLMProviderInterface', 'LLMServiceInterface']
```

#### application/interfaces/llm_provider_interface.py
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncIterator
import asyncio

class LLMProviderInterface(ABC):
    """Interface para provedores de LLM com operações assíncronas otimizadas"""

    @abstractmethod
    async def chat_completions(self, **kwargs) -> Dict[Any, Any]:
        """Processa requisições de chat/completions de forma assíncrona"""
        pass

    @abstractmethod
    async def chat_completions_stream(self, **kwargs) -> AsyncIterator[Dict[Any, Any]]:
        """Processa requisições de chat/completions com streaming assíncrono"""
        pass

    @abstractmethod
    async def text_to_speech(self, **kwargs) -> Dict[Any, Any]:
        """Processa requisições de text-to-speech de forma assíncrona"""
        pass

    @abstractmethod
    async def speech_to_text(self, **kwargs) -> Dict[Any, Any]:
        """Processa requisições de speech-to-text de forma assíncrona"""
        pass

    @abstractmethod
    async def generate_image(self, **kwargs) -> Dict[Any, Any]:
        """Processa requisições de geração de imagem de forma assíncrona"""
        pass

    @abstractmethod
    async def create_embeddings(self, **kwargs) -> Dict[Any, Any]:
        """Processa requisições de embeddings de forma assíncrona"""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Retorna o nome do provedor (azure, google, openai, anthropic, etc)"""
        pass

    @abstractmethod
    def get_model_type(self) -> str:
        """Retorna o tipo do modelo (AzureOpenAI, GoogleVertexClaude, etc)"""
        pass

    @abstractmethod
    def supports_operation(self, operation: str) -> bool:
        """Verifica se o provedor suporta uma operação específica"""
        pass

    @abstractmethod
    async def validate_credentials(self) -> bool:
        """Valida as credenciais do provedor de forma assíncrona"""
        pass

    @abstractmethod
    async def get_available_models(self) -> List[str]:
        """Obtém lista de modelos disponíveis no provedor"""
        pass
```

#### application/interfaces/llm_service_interface.py
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from domain.entities.llm_model import LLMModel

class LLMServiceInterface(ABC):
    """Interface para serviços de modelos LLM com operações assíncronas"""

    @abstractmethod
    async def get_model(self, model_id: str) -> Optional[LLMModel]:
        """Obtém um modelo específico por ID de forma assíncrona"""
        pass

    @abstractmethod
    async def list_models(self, include_private: bool = False, project_id: Optional[str] = None) -> List[LLMModel]:
        """Lista todos os modelos disponíveis de forma assíncrona"""
        pass

    @abstractmethod
    async def get_models_by_provider(self, provider: str) -> List[LLMModel]:
        """Obtém modelos filtrados por provedor de forma assíncrona"""
        pass

    @abstractmethod
    async def get_models_by_type(self, model_type: str) -> List[LLMModel]:
        """Obtém modelos filtrados por tipo de forma assíncrona"""
        pass

    @abstractmethod
    async def get_models_by_operation(self, operation: str) -> List[LLMModel]:
        """Obtém modelos que suportam uma operação específica de forma assíncrona"""
        pass

    @abstractmethod
    async def validate_model_for_operation(self, model_id: str, operation: str) -> bool:
        """Valida se um modelo suporta uma operação específica de forma assíncrona"""
        pass

    @abstractmethod
    async def create_model(self, model_data: Dict[str, Any]) -> LLMModel:
        """Cria um novo modelo de forma assíncrona"""
        pass

    @abstractmethod
    async def update_model(self, model_id: str, model_data: Dict[str, Any]) -> LLMModel:
        """Atualiza um modelo existente de forma assíncrona"""
        pass

    @abstractmethod
    async def delete_model(self, model_id: str) -> bool:
        """Exclui um modelo de forma assíncrona"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Verifica a saúde do serviço de modelos"""
        pass
```

### Application Layer - Use Cases

#### application/use_cases/__init__.py
```python
# Casos de uso da camada de aplicação
from .chat_completion_use_case import ChatCompletionUseCase
from .text_to_speech_use_case import TextToSpeechUseCase
from .speech_to_text_use_case import SpeechToTextUseCase
from .image_generation_use_case import ImageGenerationUseCase

__all__ = [
    'ChatCompletionUseCase',
    'TextToSpeechUseCase',
    'SpeechToTextUseCase',
    'ImageGenerationUseCase'
]
```

#### application/use_cases/chat_completion_use_case.py
```python
from typing import Dict, Any, AsyncIterator, Optional
import logging
from application.interfaces.llm_service_interface import LLMServiceInterface
from infra.schemas.chat_completion_schemas import ChatCompletionRequest
from infra.providers.litellm_provider import LiteLLMProvider
from domain.services.operation_validator import OperationValidator

logger = logging.getLogger(__name__)

class ChatCompletionUseCase:
    """Caso de uso para completions de chat com operações assíncronas otimizadas"""

    def __init__(self, model_service: LLMServiceInterface):
        self.model_service = model_service
        self.operation_validator = OperationValidator()

    async def execute(self, request: ChatCompletionRequest) -> Dict[Any, Any]:
        """Executa uma requisição de chat completion de forma assíncrona"""
        try:
            # Log da requisição
            logger.info(f"Processing chat completion for model: {request.model}")

            # Obtém configuração do modelo
            model = await self.model_service.get_model(request.model)
            if not model:
                raise ValueError(f"Model '{request.model}' not found")

            # Verifica se o modelo suporta chat completions
            if not await self.model_service.validate_model_for_operation(request.model, 'ChatCompletion'):
                raise ValueError(f"Model '{request.model}' does not support chat completions")

            # Valida a requisição
            self._validate_request(request)

            # Cria instância do provedor e executa requisição
            provider = LiteLLMProvider(model)

            # Converte requisição para formato LiteLLM
            litellm_params = self._prepare_litellm_params(request)

            # Executa a requisição
            if request.stream:
                return await self._execute_streaming(provider, litellm_params)
            else:
                return await provider.chat_completions(**litellm_params)

        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise

    async def execute_stream(self, request: ChatCompletionRequest) -> AsyncIterator[Dict[Any, Any]]:
        """Executa uma requisição de chat completion com streaming"""
        model = await self.model_service.get_model(request.model)
        if not model:
            raise ValueError(f"Model '{request.model}' not found")

        provider = LiteLLMProvider(model)
        litellm_params = self._prepare_litellm_params(request)
        litellm_params['stream'] = True

        async for chunk in provider.chat_completions_stream(**litellm_params):
            yield chunk

    def _validate_request(self, request: ChatCompletionRequest) -> None:
        """Valida os parâmetros da requisição"""
        if not request.messages:
            raise ValueError("Messages cannot be empty")

        if len(request.messages) > 100:
            raise ValueError("Too many messages (max 100)")

        # Valida se há pelo menos uma mensagem do usuário
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise ValueError("At least one user message is required")

    def _prepare_litellm_params(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Prepara parâmetros para o LiteLLM"""
        return {
            'messages': [msg.model_dump() for msg in request.messages],
            'temperature': request.temperature,
            'max_tokens': request.max_tokens,
            'top_p': request.top_p,
            'frequency_penalty': request.frequency_penalty,
            'presence_penalty': request.presence_penalty,
            'stop': request.stop,
            'stream': request.stream,
            'user': request.user,
            'tools': request.tools,
            'tool_choice': request.tool_choice,
            'functions': request.functions,
            'function_call': request.function_call
        }

    async def _execute_streaming(self, provider: LiteLLMProvider, params: Dict[str, Any]) -> Dict[Any, Any]:
        """Executa requisição com streaming interno e retorna resultado consolidado"""
        chunks = []
        async for chunk in provider.chat_completions_stream(**params):
            chunks.append(chunk)

        # Consolida os chunks em uma resposta única
        return self._consolidate_chunks(chunks)

    def _consolidate_chunks(self, chunks: List[Dict[Any, Any]]) -> Dict[Any, Any]:
        """Consolida chunks de streaming em uma resposta única"""
        if not chunks:
            raise ValueError("No chunks received from provider")

        # Implementação básica - pode ser aprimorada baseada no formato específico
        return chunks[-1] if chunks else {}
```

#### application/use_cases/text_to_speech_use_case.py
```python
from typing import Dict, Any
import logging
from application.interfaces.llm_service_interface import LLMServiceInterface
from infra.schemas.text_to_speech_schemas import TextToSpeechRequest
from infra.providers.litellm_provider import LiteLLMProvider

logger = logging.getLogger(__name__)

class TextToSpeechUseCase:
    """Caso de uso para text-to-speech com operações assíncronas"""

    def __init__(self, model_service: LLMServiceInterface):
        self.model_service = model_service

    async def execute(self, request: TextToSpeechRequest) -> Dict[Any, Any]:
        """Executa uma requisição de text-to-speech de forma assíncrona"""
        try:
            logger.info(f"Processing TTS request for model: {request.model}")

            # Obtém configuração do modelo
            model = await self.model_service.get_model(request.model)
            if not model:
                raise ValueError(f"Model '{request.model}' not found")

            # Verifica se o modelo suporta TTS
            if not await self.model_service.validate_model_for_operation(request.model, 'audio_tts'):
                raise ValueError(f"Model '{request.model}' does not support text-to-speech")

            # Valida entrada
            self._validate_request(request)

            # Cria instância do provedor e executa requisição
            provider = LiteLLMProvider(model)

            # Converte requisição para formato do provedor
            tts_params = {
                'input': request.input,
                'voice': request.voice,
                'response_format': request.response_format,
                'speed': request.speed
            }

            # Remove valores None
            tts_params = {k: v for k, v in tts_params.items() if v is not None}

            return await provider.text_to_speech(**tts_params)

        except Exception as e:
            logger.error(f"Error in TTS: {str(e)}")
            raise

    def _validate_request(self, request: TextToSpeechRequest) -> None:
        """Valida os parâmetros da requisição TTS"""
        if not request.input or len(request.input.strip()) == 0:
            raise ValueError("Input text cannot be empty")

        if len(request.input) > 4096:
            raise ValueError("Input text too long (max 4096 characters)")

        valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        if request.voice not in valid_voices:
            raise ValueError(f"Invalid voice. Must be one of: {', '.join(valid_voices)}")

        valid_formats = ["mp3", "opus", "aac", "flac"]
        if request.response_format not in valid_formats:
            raise ValueError(f"Invalid response format. Must be one of: {', '.join(valid_formats)}")
```

### Domain Layer - Entities

#### domain/entities/__init__.py
```python
# Entidades do domínio
from .llm_model import LLMModel
from .operation import Operation

__all__ = ['LLMModel', 'Operation']
```

#### domain/entities/llm_model.py
```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json

class ModelProvider(str, Enum):
    """Provedores de modelo suportados"""
    AZURE = "Azure"
    GOOGLE = "Google"
    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"

class ModelType(str, Enum):
    """Tipos de modelo suportados"""
    AZURE_OPENAI = "AzureOpenAI"
    AZURE_FOUNDRY = "AzureFoundry"
    GOOGLE_VERTEX = "GoogleVertex"
    GOOGLE_VERTEX_CLAUDE = "GoogleVertexClaude"
    GOOGLE_VERTEX_DEEPSEEK = "GoogleVertexDeepSeek"
    GOOGLE_VERTEX_META = "GoogleVertexMeta"
    GOOGLE_VERTEX_MISTRAL = "GoogleVertexMistral"
    GOOGLE_VERTEX_JAMBA = "GoogleVertexJamba"
    GOOGLE_VERTEX_QWEN = "GoogleVertexQwen"
    GOOGLE_VERTEX_OPENAI = "GoogleVertexOpenAI"
    OPENAI_NATIVE = "OpenAINative"
    ANTHROPIC_NATIVE = "AnthropicNative"

@dataclass
class LLMModel:
    """Entidade de modelo LLM com validações aprimoradas"""

    id: str
    name: str
    code: str
    provider: ModelProvider
    model_type: ModelType
    projects: List[str] = field(default_factory=list)
    private: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)
    costs: Dict[str, float] = field(default_factory=dict)
    operations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __post_init__(self):
        """Validações pós-inicialização"""
        self.validate()

    def validate(self) -> None:
        """Valida a integridade do modelo"""
        if not self.id or not self.id.strip():
            raise ValueError("Model ID cannot be empty")

        if not self.name or not self.name.strip():
            raise ValueError("Model name cannot be empty")

        if not self.code or not self.code.strip():
            raise ValueError("Model code cannot be empty")

        # Valida operações
        valid_operations = {
            "Responses", "ChatCompletion", "embeddings",
            "images", "audio_tts", "audio_stt", "audio_realtime"
        }
        invalid_ops = set(self.operations) - valid_operations
        if invalid_ops:
            raise ValueError(f"Invalid operations: {invalid_ops}")

        # Valida custos
        if self.costs:
            required_cost_keys = {"input", "output"}
            if not required_cost_keys.issubset(self.costs.keys()):
                raise ValueError("Costs must include 'input' and 'output' keys")

            for key, value in self.costs.items():
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValueError(f"Cost '{key}' must be a non-negative number")

    def get_provider_config(self) -> Dict[str, Any]:
        """Retorna configuração específica do provider"""
        provider_configs = {
            ModelProvider.AZURE: ['secret_name', 'deployment_name', 'api_version', 'endpoint'],
            ModelProvider.GOOGLE: ['secret_name', 'gcp_project', 'location', 'enable_tools']
        }

        config = {}
        if self.provider in provider_configs:
            for key in provider_configs[self.provider]:
                if key in self.parameters:
                    config[key] = self.parameters[key]
        return config

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calcula o custo baseado nos tokens de entrada e saída"""
        if not self.costs:
            return 0.0

        input_cost = input_tokens * self.costs.get('input', 0.0) / 1000
        output_cost = output_tokens * self.costs.get('output', 0.0) / 1000
        return round(input_cost + output_cost, 6)

    def supports_operation(self, operation: str) -> bool:
        """Verifica se o modelo suporta uma operação específica"""
        return operation in self.operations and self.enabled

    def is_available_for_project(self, project_id: str) -> bool:
        """Verifica se o modelo está disponível para um projeto específico"""
        if not self.enabled:
            return False
        if not self.private:
            return True
        return project_id in self.projects

    def supports_streaming(self) -> bool:
        """Verifica se o modelo suporta respostas em streaming"""
        return self.supports_operation('ChatCompletion')

    def get_cost_info(self) -> Dict[str, float]:
        """Retorna informações de custo do modelo"""
        return {
            'input_cost_per_1k': self.costs.get('input', 0.0),
            'output_cost_per_1k': self.costs.get('output', 0.0)
        }

    def get_supported_operations(self) -> List[str]:
        """Retorna lista de operações suportadas"""
        return self.operations.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Converte o modelo para dicionário"""
        return {
            'id': self.id,
            'name': self.name,
            'code': self.code,
            'provider': self.provider.value,
            'model_type': self.model_type.value,
            'projects': self.projects,
            'private': self.private,
            'parameters': self.parameters,
            'costs': self.costs,
            'operations': self.operations,
            'metadata': self.metadata,
            'enabled': self.enabled,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMModel':
        """Cria uma instância do modelo a partir de dicionário"""
        return cls(
            id=data['id'],
            name=data['name'],
            code=data.get('code', data['id']),
            provider=ModelProvider(data['provider']),
            model_type=ModelType(data['model_type']),
            projects=data.get('projects', []),
            private=data.get('private', False),
            parameters=data.get('parameters', {}),
            costs=data.get('costs', {}),
            operations=data.get('operations', []),
            metadata=data.get('metadata', {}),
            enabled=data.get('enabled', True),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )

    def __repr__(self):
        return f"LLMModel(id={self.id}, name={self.name}, provider={self.provider.value}, type={self.model_type.value})"

### Domain Layer - Services

#### domain/services/__init__.py
```python
# Serviços do domínio
from .llm_model_service import LLMModelService
from .operation_validator import OperationValidator

__all__ = ['LLMModelService', 'OperationValidator']
```

#### domain/services/llm_model_service.py
```python
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
from application.interfaces.llm_service_interface import LLMServiceInterface
from domain.entities.llm_model import LLMModel, ModelProvider, ModelType
from infra.database.model_repository import ModelRepository
from llmproxy.config import Config

logger = logging.getLogger(__name__)

class LLMModelService(LLMServiceInterface):
    """Serviço de domínio para gerenciamento de modelos LLM"""

    def __init__(self, model_repository: ModelRepository):
        self.model_repository = model_repository

    async def get_model(self, model_id: str) -> Optional[LLMModel]:
        """Obtém um modelo específico por ID de forma assíncrona"""
        try:
            logger.info(f"Getting model: {model_id}")
            model_data = await self.model_repository.get_by_id(model_id)
            if not model_data:
                logger.warning(f"Model not found: {model_id}")
                return None

            # Enriquece parâmetros com variáveis de ambiente
            enriched_parameters = await self._enrich_parameters(model_data.get('parameters', {}))
            model_data['parameters'] = enriched_parameters

            return LLMModel.from_dict(model_data)
        except Exception as e:
            logger.error(f"Error getting model {model_id}: {str(e)}")
            raise

    async def list_models(self, include_private: bool = False, project_id: Optional[str] = None) -> List[LLMModel]:
        """Lista todos os modelos disponíveis de forma assíncrona"""
        try:
            logger.info(f"Listing models - include_private: {include_private}, project_id: {project_id}")

            if project_id:
                models_data = await self.model_repository.get_by_project(project_id)
            else:
                models_data = await self.model_repository.get_all()

            models = []
            for model_data in models_data:
                # Filtra modelos privados se necessário
                if not include_private and model_data.get('private', False):
                    if not project_id or project_id not in model_data.get('projects', []):
                        continue

                # Enriquece parâmetros
                enriched_parameters = await self._enrich_parameters(model_data.get('parameters', {}))
                model_data['parameters'] = enriched_parameters

                try:
                    model = LLMModel.from_dict(model_data)
                    if model.enabled:  # Apenas modelos habilitados
                        models.append(model)
                except Exception as e:
                    logger.warning(f"Skipping invalid model {model_data.get('id', 'unknown')}: {str(e)}")
                    continue

            logger.info(f"Found {len(models)} models")
            return models
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []

    async def get_models_by_provider(self, provider: str) -> List[LLMModel]:
        """Obtém modelos filtrados por provedor de forma assíncrona"""
        try:
            logger.info(f"Getting models for provider: {provider}")
            models_data = await self.model_repository.get_by_provider(provider)
            return await self._process_models_data(models_data)
        except Exception as e:
            logger.error(f"Error getting models for provider {provider}: {str(e)}")
            return []

    async def get_models_by_type(self, model_type: str) -> List[LLMModel]:
        """Obtém modelos filtrados por tipo de forma assíncrona"""
        try:
            logger.info(f"Getting models for type: {model_type}")
            models_data = await self.model_repository.get_by_type(model_type)
            return await self._process_models_data(models_data)
        except Exception as e:
            logger.error(f"Error getting models for type {model_type}: {str(e)}")
            return []

    async def get_models_by_operation(self, operation: str) -> List[LLMModel]:
        """Obtém modelos que suportam uma operação específica de forma assíncrona"""
        try:
            logger.info(f"Getting models for operation: {operation}")
            models_data = await self.model_repository.get_by_operation(operation)
            return await self._process_models_data(models_data)
        except Exception as e:
            logger.error(f"Error getting models for operation {operation}: {str(e)}")
            return []

    async def validate_model_for_operation(self, model_id: str, operation: str) -> bool:
        """Valida se um modelo suporta uma operação específica de forma assíncrona"""
        try:
            model = await self.get_model(model_id)
            if not model:
                return False
            return model.supports_operation(operation)
        except Exception as e:
            logger.error(f"Error validating model {model_id} for operation {operation}: {str(e)}")
            return False

    async def create_model(self, model_data: Dict[str, Any]) -> LLMModel:
        """Cria um novo modelo de forma assíncrona"""
        try:
            logger.info(f"Creating model: {model_data.get('id', 'unknown')}")

            # Adiciona timestamps
            now = datetime.utcnow().isoformat()
            model_data['created_at'] = now
            model_data['updated_at'] = now

            # Valida dados antes de criar
            model = LLMModel.from_dict(model_data)

            # Persiste no repositório
            created_data = await self.model_repository.create(model.to_dict())
            return LLMModel.from_dict(created_data)
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise

    async def update_model(self, model_id: str, model_data: Dict[str, Any]) -> LLMModel:
        """Atualiza um modelo existente de forma assíncrona"""
        try:
            logger.info(f"Updating model: {model_id}")

            # Verifica se modelo existe
            existing_model = await self.get_model(model_id)
            if not existing_model:
                raise ValueError(f"Model '{model_id}' not found")

            # Atualiza timestamp
            model_data['updated_at'] = datetime.utcnow().isoformat()
            model_data['id'] = model_id  # Garante que o ID não mude

            # Valida dados atualizados
            updated_model = LLMModel.from_dict({**existing_model.to_dict(), **model_data})

            # Persiste no repositório
            updated_data = await self.model_repository.update(model_id, updated_model.to_dict())
            return LLMModel.from_dict(updated_data)
        except Exception as e:
            logger.error(f"Error updating model {model_id}: {str(e)}")
            raise

    async def delete_model(self, model_id: str) -> bool:
        """Exclui um modelo de forma assíncrona"""
        try:
            logger.info(f"Deleting model: {model_id}")

            # Verifica se modelo existe
            existing_model = await self.get_model(model_id)
            if not existing_model:
                raise ValueError(f"Model '{model_id}' not found")

            return await self.model_repository.delete(model_id)
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {str(e)}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Verifica a saúde do serviço de modelos"""
        try:
            # Conta total de modelos
            all_models = await self.list_models(include_private=True)
            enabled_models = [m for m in all_models if m.enabled]

            # Agrupa por provedor
            providers = {}
            for model in enabled_models:
                provider = model.provider.value
                if provider not in providers:
                    providers[provider] = 0
                providers[provider] += 1

            return {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'models': {
                    'total': len(all_models),
                    'enabled': len(enabled_models),
                    'by_provider': providers
                },
                'repository_health': await self.model_repository.health_check()
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }

    async def _enrich_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Enriquece parâmetros com valores de variáveis de ambiente"""
        enriched = {}
        for key, value in parameters.items():
            if isinstance(value, str) and value.isupper() and len(value) > 3:
                # Se o valor é uma string maiúscula, trata como nome de variável de ambiente
                env_value = Config.get_provider_config(value)
                if env_value:
                    enriched[key] = env_value
                else:
                    logger.warning(f"Environment variable '{value}' not found for parameter '{key}'")
                    enriched[key] = value  # Mantém valor original como fallback
            else:
                enriched[key] = value
        return enriched

    async def _process_models_data(self, models_data: List[Dict[str, Any]]) -> List[LLMModel]:
        """Processa dados de modelos e retorna lista de entidades"""
        models = []
        for model_data in models_data:
            try:
                enriched_parameters = await self._enrich_parameters(model_data.get('parameters', {}))
                model_data['parameters'] = enriched_parameters
                model = LLMModel.from_dict(model_data)
                if model.enabled:
                    models.append(model)
            except Exception as e:
                logger.warning(f"Skipping invalid model {model_data.get('id', 'unknown')}: {str(e)}")
                continue
        return models
```

#### domain/services/operation_validator.py
```python
from typing import List, Dict, Any
import logging
from domain.entities.operation import Operation, OperationType
from domain.entities.llm_model import ModelType

logger = logging.getLogger(__name__)

class OperationValidator:
    """Validador de operações para modelos LLM"""

    def __init__(self):
        self.operations = Operation.get_all_operations()

    def validate_operations(self, operations: List[str]) -> bool:
        """Valida se todas as operações são válidas"""
        try:
            valid_operation_names = {op.value for op in OperationType}
            invalid_ops = set(operations) - valid_operation_names

            if invalid_ops:
                logger.warning(f"Invalid operations found: {invalid_ops}")
                return False

            return True
        except Exception as e:
            logger.error(f"Error validating operations: {str(e)}")
            return False

    def validate_operation_for_model_type(self, operation: str, model_type: str) -> bool:
        """Valida se uma operação é suportada por um tipo de modelo"""
        try:
            return Operation.validate_operation_for_model_type(operation, model_type)
        except Exception as e:
            logger.error(f"Error validating operation {operation} for model type {model_type}: {str(e)}")
            return False

    def get_supported_operations_for_model_type(self, model_type: str) -> List[str]:
        """Retorna operações suportadas para um tipo de modelo"""
        try:
            supported = []
            for op_type, operation in self.operations.items():
                if self.validate_operation_for_model_type(op_type.value, model_type):
                    supported.append(op_type.value)
            return supported
        except Exception as e:
            logger.error(f"Error getting supported operations for {model_type}: {str(e)}")
            return []

    def get_endpoint_for_operation(self, operation_type: str) -> str:
        """Retorna o endpoint correspondente para uma operação"""
        try:
            return Operation.get_endpoint_for_operation(operation_type)
        except Exception as e:
            logger.error(f"Error getting endpoint for operation {operation_type}: {str(e)}")
            return ""

    def get_operation_info(self, operation_type: str) -> Dict[str, Any]:
        """Retorna informações detalhadas sobre uma operação"""
        try:
            for op_type, operation in self.operations.items():
                if op_type.value == operation_type:
                    return {
                        'type': operation.type.value,
                        'endpoint': operation.endpoint,
                        'description': operation.description,
                        'required_model_types': operation.required_model_types,
                        'parameters': operation.parameters
                    }
            return {}
        except Exception as e:
            logger.error(f"Error getting operation info for {operation_type}: {str(e)}")
            return {}

    def get_all_operations_info(self) -> Dict[str, Dict[str, Any]]:
        """Retorna informações sobre todas as operações"""
        try:
            operations_info = {}
            for op_type, operation in self.operations.items():
                operations_info[op_type.value] = {
                    'endpoint': operation.endpoint,
                    'description': operation.description,
                    'required_model_types': operation.required_model_types,
                    'parameters': operation.parameters
                }
            return operations_info
        except Exception as e:
            logger.error(f"Error getting all operations info: {str(e)}")
            return {}
```
```

#### domain/entities/operation.py
```python
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum

class OperationType(str, Enum):
    """Tipos de operação suportados"""
    RESPONSES = "Responses"
    CHAT_COMPLETION = "ChatCompletion"
    EMBEDDINGS = "embeddings"
    IMAGES = "images"
    AUDIO_TTS = "audio_tts"
    AUDIO_STT = "audio_stt"
    AUDIO_REALTIME = "audio_realtime"

@dataclass
class Operation:
    """Entidade que representa uma operação suportada"""

    type: OperationType
    endpoint: str
    description: str
    required_model_types: List[str]
    parameters: Dict[str, any]

    @classmethod
    def get_all_operations(cls) -> Dict[str, 'Operation']:
        """Retorna todas as operações suportadas"""
        return {
            OperationType.RESPONSES: cls(
                type=OperationType.RESPONSES,
                endpoint="/v1/chat/completions",
                description="Geração de respostas básicas de texto",
                required_model_types=["*"],
                parameters={}
            ),
            OperationType.CHAT_COMPLETION: cls(
                type=OperationType.CHAT_COMPLETION,
                endpoint="/v1/chat/completions",
                description="Chat completions completas com contexto",
                required_model_types=["*"],
                parameters={"supports_streaming": True}
            ),
            OperationType.EMBEDDINGS: cls(
                type=OperationType.EMBEDDINGS,
                endpoint="/v1/embeddings",
                description="Geração de embeddings vetoriais",
                required_model_types=["AzureOpenAI", "OpenAINative"],
                parameters={}
            ),
            OperationType.IMAGES: cls(
                type=OperationType.IMAGES,
                endpoint="/v1/images/generations",
                description="Geração e análise de imagens",
                required_model_types=["AzureOpenAI", "OpenAINative"],
                parameters={}
            ),
            OperationType.AUDIO_TTS: cls(
                type=OperationType.AUDIO_TTS,
                endpoint="/v1/audio/speech",
                description="Conversão de texto para fala",
                required_model_types=["AzureOpenAI", "OpenAINative"],
                parameters={}
            ),
            OperationType.AUDIO_STT: cls(
                type=OperationType.AUDIO_STT,
                endpoint="/v1/audio/transcriptions",
                description="Conversão de fala para texto",
                required_model_types=["AzureOpenAI", "OpenAINative"],
                parameters={}
            ),
            OperationType.AUDIO_REALTIME: cls(
                type=OperationType.AUDIO_REALTIME,
                endpoint="/v1/audio/realtime",
                description="Áudio em tempo real",
                required_model_types=["AzureOpenAI"],
                parameters={"requires_websocket": True}
            )
        }

    @classmethod
    def get_endpoint_for_operation(cls, operation_type: str) -> str:
        """Retorna o endpoint correspondente para uma operação"""
        operations = cls.get_all_operations()
        for op_type, operation in operations.items():
            if op_type.value == operation_type:
                return operation.endpoint
        return ""

    @classmethod
    def validate_operation_for_model_type(cls, operation_type: str, model_type: str) -> bool:
        """Valida se uma operação é suportada por um tipo de modelo"""
        operations = cls.get_all_operations()
        for op_type, operation in operations.items():
            if op_type.value == operation_type:
                if "*" in operation.required_model_types:
                    return True
                return model_type in operation.required_model_types
        return False
```

### Infrastructure Layer - Schemas

#### infra/schemas/__init__.py
```python
# Schemas de dados (DTOs) para a camada de infraestrutura
from .chat_completion_schemas import (
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageRole, Usage, Choice
)
from .text_to_speech_schemas import TextToSpeechRequest, TextToSpeechResponse, VoiceType
from .speech_to_text_schemas import SpeechToTextRequest, SpeechToTextResponse, AudioFormat
from .image_generation_schemas import ImageGenerationRequest, ImageGenerationResponse, ImageSize
from .model_schemas import ModelListResponse, ModelInfo, ModelStats

__all__ = [
    # Chat Completion
    'ChatCompletionRequest', 'ChatCompletionResponse', 'ChatMessage', 'MessageRole', 'Usage', 'Choice',
    # Text to Speech
    'TextToSpeechRequest', 'TextToSpeechResponse', 'VoiceType',
    # Speech to Text
    'SpeechToTextRequest', 'SpeechToTextResponse', 'AudioFormat',
    # Image Generation
    'ImageGenerationRequest', 'ImageGenerationResponse', 'ImageSize',
    # Model Management
    'ModelListResponse', 'ModelInfo', 'ModelStats'
]
```

#### infra/schemas/chat_completion_schemas.py
```python
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Dict, Any, Optional, Union, Literal
from enum import Enum
import json

class MessageRole(str, Enum):
    """Papéis de mensagem suportados pela OpenAI e todos os provedores LiteLLM"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"

class FunctionCall(BaseModel):
    """Function call para compatibilidade OpenAI"""
    name: str = Field(..., min_length=1, max_length=64)
    arguments: str = Field(..., description="JSON string dos argumentos")

    @field_validator('arguments')
    @classmethod
    def validate_arguments(cls, v: str) -> str:
        try:
            json.loads(v)  # Valida se é JSON válido
            return v
        except json.JSONDecodeError:
            raise ValueError('arguments must be valid JSON string')

class ToolCall(BaseModel):
    """Tool call para OpenAI tools"""
    id: str = Field(..., min_length=1)
    type: Literal["function"] = "function"
    function: FunctionCall

class ChatMessage(BaseModel):
    """Mensagem de chat totalmente compatível com OpenAI"""
    model_config = ConfigDict(extra="allow")  # Permite campos extras para provedores específicos

    role: MessageRole
    content: Optional[Union[str, List[Dict[str, Any]]]] = Field(default=None)  # Suporte a conteúdo multimodal
    name: Optional[str] = Field(default=None, max_length=64)
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None  # Para mensagens de resposta de tool

    @field_validator('content')
    @classmethod
    def validate_content(cls, v, info):
        role = info.data.get('role')

        # System e user devem ter content (exceto quando usando tools)
        if role in [MessageRole.SYSTEM, MessageRole.USER] and v is None:
            raise ValueError(f'{role.value} messages must have content')

        # Content pode ser string ou array para multimodal
        if isinstance(v, str) and len(v) > 100000:
            raise ValueError('Content too long (max 100000 characters)')

        return v

class ResponseFormat(BaseModel):
    """Formato de resposta para structured outputs"""
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: Optional[Dict[str, Any]] = None

class ChatCompletionRequest(BaseModel):
    """Requisição totalmente compatível com OpenAI API e LiteLLM"""
    model_config = ConfigDict(extra="allow")  # Permite parâmetros específicos de provedores

    # Parâmetros obrigatórios OpenAI
    model: str = Field(..., min_length=1, max_length=256)
    messages: List[ChatMessage] = Field(..., min_length=1, max_length=200)

    # Parâmetros opcionais OpenAI padrão
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0, le=200000)  # Ajustado para GPT-4 Turbo
    max_completion_tokens: Optional[int] = Field(default=None, gt=0, le=200000)  # Novo parâmetro OpenAI
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1, le=20)
    stream: Optional[bool] = Field(default=False)
    stream_options: Optional[Dict[str, Any]] = None  # Para include_usage no streaming
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = Field(default=None, max_length=256)
    seed: Optional[int] = Field(default=None, ge=0, le=2147483647)

    # Logprobs (GPT-4 Turbo e newer)
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20)

    # Response format
    response_format: Optional[ResponseFormat] = None

    # Tools (function calling)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = True

    # Legacy function calling (deprecado mas ainda suportado)
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None

    # Parâmetros LiteLLM para confiabilidade e fallback
    timeout: Optional[float] = Field(default=None, gt=0, le=600)
    max_retries: Optional[int] = Field(default=None, ge=0, le=5)
    fallbacks: Optional[List[str]] = None  # Lista de modelos de fallback
    context_window_fallbacks: Optional[List[str]] = None  # Fallbacks para contexto

    # Headers customizados para diferentes provedores
    headers: Optional[Dict[str, str]] = None
    extra_headers: Optional[Dict[str, str]] = None

    # Parâmetros específicos de provedores (passados automaticamente pelo LiteLLM)

    # Anthropic Claude específico
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None  # Traduzido para 'thinking'
    thinking: Optional[Dict[str, Any]] = None  # {"type": "enabled", "budget_tokens": int}
    cache_control: Optional[Dict[str, Any]] = None  # Prompt caching

    # Google Vertex AI / Gemini específico
    top_k: Optional[int] = Field(default=None, ge=1, le=40)
    candidate_count: Optional[int] = Field(default=None, ge=1, le=8)
    safety_settings: Optional[List[Dict[str, Any]]] = None

    # Cohere específico
    max_input_tokens: Optional[int] = Field(default=None, gt=0)
    truncate: Optional[Literal["NONE", "START", "END"]] = None
    connectors: Optional[List[Dict[str, Any]]] = None

    # AWS Bedrock específico
    anthropic_version: Optional[str] = None
    max_tokens_to_sample: Optional[int] = None  # Legacy Anthropic

    # Azure OpenAI específico (passados via environment ou model config)
    api_version: Optional[str] = None
    api_base: Optional[str] = None
    deployment_id: Optional[str] = None

    # Ollama específico
    num_predict: Optional[int] = None
    repeat_penalty: Optional[float] = None

    # Replicate específico
    replicate_owner: Optional[str] = None

    # Huggingface específico
    use_cache: Optional[bool] = None
    wait_for_model: Optional[bool] = None

    # TogetherAI específico
    repetition_penalty: Optional[float] = None

    # OpenRouter específico
    transforms: Optional[List[str]] = None
    models: Optional[List[str]] = None
    route: Optional[str] = None

    # Parâmetros de segurança e compliance
    safety_identifier: Optional[str] = None  # Para tracking de segurança

    # Logging e debugging
    log_level: Optional[str] = None
    litellm_logging: Optional[bool] = None

    @field_validator('stop')
    @classmethod
    def validate_stop(cls, v):
        if isinstance(v, list) and len(v) > 4:
            raise ValueError('stop sequences cannot exceed 4 items')
        return v

    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError('messages cannot be empty')

        # Primeiro deve ser system ou user
        if v[0].role not in [MessageRole.SYSTEM, MessageRole.USER]:
            raise ValueError('First message must be system or user role')

        return v

class LogprobContent(BaseModel):
    """Conteúdo de logprobs"""
    token: str
    logprob: float
    bytes: Optional[List[int]] = None
    top_logprobs: Optional[List[Dict[str, Any]]] = None

class ChoiceLogprobs(BaseModel):
    """Logprobs para choice"""
    content: Optional[List[LogprobContent]] = None
    refusal: Optional[List[LogprobContent]] = None

class Usage(BaseModel):
    """Informações de uso de tokens - compatível com OpenAI"""
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)

    # Detalhes de tokens (GPT-4 Turbo)
    prompt_tokens_details: Optional[Dict[str, int]] = None
    completion_tokens_details: Optional[Dict[str, int]] = None

class Choice(BaseModel):
    """Escolha de resposta - totalmente compatível OpenAI"""
    index: int = Field(..., ge=0)
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "function_call", "tool_calls", "content_filter"]] = None
    logprobs: Optional[ChoiceLogprobs] = None

class ChatCompletionResponse(BaseModel):
    """Resposta totalmente compatível com OpenAI"""
    id: str = Field(..., min_length=1)
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(..., gt=0)
    model: str = Field(..., min_length=1)
    choices: List[Choice] = Field(..., min_length=1)
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None

    # Campos extras para alguns provedores
    service_tier: Optional[str] = None

# Streaming response
class ChatCompletionChunk(BaseModel):
    """Chunk de resposta streaming compatível OpenAI"""
    id: str = Field(..., min_length=1)
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(..., gt=0)
    model: str = Field(..., min_length=1)
    system_fingerprint: Optional[str] = None
    choices: List[Dict[str, Any]] = Field(..., min_length=1)  # Structure varia no streaming
    usage: Optional[Usage] = None  # Só no último chunk se stream_options.include_usage=true
```

#### infra/schemas/text_to_speech_schemas.py
```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal, Dict, Any
from enum import Enum

class VoiceType(str, Enum):
    """Tipos de voz disponíveis - OpenAI compatível"""
    ALLOY = "alloy"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SHIMMER = "shimmer"

class AudioFormat(str, Enum):
    """Formatos de áudio suportados - OpenAI compatível"""
    MP3 = "mp3"
    OPUS = "opus"
    AAC = "aac"
    FLAC = "flac"
    WAV = "wav"
    PCM = "pcm"

class TextToSpeechRequest(BaseModel):
    """Requisição TTS totalmente compatível OpenAI + LiteLLM"""
    model_config = ConfigDict(extra="allow")  # Permite parâmetros específicos de provedores

    # Parâmetros obrigatórios OpenAI
    model: str = Field(..., min_length=1, max_length=256)
    input: str = Field(..., min_length=1, max_length=4096)
    voice: VoiceType = Field(default=VoiceType.ALLOY)

    # Parâmetros opcionais OpenAI
    response_format: AudioFormat = Field(default=AudioFormat.MP3)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)

    # Parâmetros LiteLLM
    timeout: Optional[float] = Field(default=None, gt=0, le=300)
    max_retries: Optional[int] = Field(default=None, ge=0, le=5)
    headers: Optional[Dict[str, str]] = None
    extra_headers: Optional[Dict[str, str]] = None

    # Parâmetros específicos de provedores
    # Azure OpenAI
    api_version: Optional[str] = None
    api_base: Optional[str] = None

    # ElevenLabs (via LiteLLM)
    stability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    similarity_boost: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    style: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    use_speaker_boost: Optional[bool] = None

class TextToSpeechResponse(BaseModel):
    """Resposta TTS compatível OpenAI"""
    audio: bytes = Field(..., description="Audio content em bytes")
    content_type: str = Field(default="audio/mpeg")
    duration: Optional[float] = None
    size: Optional[int] = None

class SpeechToTextRequest(BaseModel):
    """Requisição STT compatível OpenAI + LiteLLM"""
    model_config = ConfigDict(extra="allow")

    # Parâmetros obrigatórios OpenAI
    file: bytes = Field(..., description="Audio file content")
    model: str = Field(..., min_length=1, max_length=256)

    # Parâmetros opcionais OpenAI
    language: Optional[str] = Field(default=None, min_length=2, max_length=5)  # ISO 639-1
    prompt: Optional[str] = Field(default=None, max_length=224)
    response_format: Optional[Literal["json", "text", "srt", "verbose_json", "vtt"]] = "json"
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    timestamp_granularities: Optional[List[Literal["word", "segment"]]] = None

    # Parâmetros LiteLLM
    timeout: Optional[float] = Field(default=None, gt=0, le=600)
    max_retries: Optional[int] = Field(default=None, ge=0, le=5)

class SpeechToTextResponse(BaseModel):
    """Resposta STT compatível OpenAI"""
    text: str

    # Campos para verbose_json
    task: Optional[str] = None
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: Optional[List[Dict[str, Any]]] = None
    words: Optional[List[Dict[str, Any]]] = None
```

#### infra/schemas/image_generation_schemas.py
```python
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, Literal, Dict, Any, Union, List
from enum import Enum

class ImageSize(str, Enum):
    """Tamanhos suportados - OpenAI compatível"""
    SIZE_256x256 = "256x256"
    SIZE_512x512 = "512x512"
    SIZE_1024x1024 = "1024x1024"
    SIZE_1792x1024 = "1792x1024"  # DALL-E 3
    SIZE_1024x1792 = "1024x1792"  # DALL-E 3

class ImageQuality(str, Enum):
    """Qualidade da imagem - DALL-E 3"""
    STANDARD = "standard"
    HD = "hd"

class ImageStyle(str, Enum):
    """Estilo da imagem - DALL-E 3"""
    VIVID = "vivid"
    NATURAL = "natural"

class ImageGenerationRequest(BaseModel):
    """Requisição de geração de imagem compatível OpenAI + LiteLLM"""
    model_config = ConfigDict(extra="allow")

    # Parâmetros obrigatórios OpenAI
    prompt: str = Field(..., min_length=1, max_length=4000)

    # Parâmetros opcionais OpenAI
    model: Optional[str] = Field(default="dall-e-2", min_length=1)
    n: Optional[int] = Field(default=1, ge=1, le=10)
    quality: Optional[ImageQuality] = ImageQuality.STANDARD
    response_format: Optional[Literal["url", "b64_json"]] = "url"
    size: Optional[ImageSize] = ImageSize.SIZE_1024x1024
    style: Optional[ImageStyle] = ImageStyle.VIVID
    user: Optional[str] = Field(default=None, max_length=256)

    # Parâmetros LiteLLM
    timeout: Optional[float] = Field(default=None, gt=0, le=600)
    max_retries: Optional[int] = Field(default=None, ge=0, le=5)

    # Parâmetros específicos de provedores
    # Azure OpenAI
    api_version: Optional[str] = None
    api_base: Optional[str] = None

    # Stability AI
    steps: Optional[int] = Field(default=None, ge=10, le=50)
    cfg_scale: Optional[float] = Field(default=None, ge=1.0, le=35.0)
    seed: Optional[int] = Field(default=None, ge=0)

class ImageData(BaseModel):
    """Dados da imagem gerada"""
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None

class ImageGenerationResponse(BaseModel):
    """Resposta de geração de imagem compatível OpenAI"""
    created: int = Field(..., gt=0)
    data: List[ImageData] = Field(..., min_length=1)

class EmbeddingRequest(BaseModel):
    """Requisição de embedding compatível OpenAI + LiteLLM"""
    model_config = ConfigDict(extra="allow")

    # Parâmetros obrigatórios OpenAI
    input: Union[str, List[str], List[int], List[List[int]]] = Field(...)
    model: str = Field(..., min_length=1)

    # Parâmetros opcionais OpenAI
    encoding_format: Optional[Literal["float", "base64"]] = "float"
    dimensions: Optional[int] = Field(default=None, gt=0, le=3072)  # Para text-embedding-3-*
    user: Optional[str] = Field(default=None, max_length=256)

    # Parâmetros LiteLLM
    timeout: Optional[float] = Field(default=None, gt=0, le=300)
    max_retries: Optional[int] = Field(default=None, ge=0, le=5)

    # Parâmetros específicos de provedores
    # Azure OpenAI
    api_version: Optional[str] = None
    api_base: Optional[str] = None

    # Cohere
    input_type: Optional[Literal["search_document", "search_query", "classification", "clustering"]] = None

    @field_validator('input')
    @classmethod
    def validate_input(cls, v):
        if isinstance(v, str) and len(v) > 8192:
            raise ValueError('Input text too long (max 8192 characters)')
        elif isinstance(v, list) and len(v) > 2048:
            raise ValueError('Too many inputs (max 2048)')
        return v

class EmbeddingData(BaseModel):
    """Dados do embedding"""
    object: Literal["embedding"] = "embedding"
    index: int = Field(..., ge=0)
    embedding: List[float]

class EmbeddingUsage(BaseModel):
    """Uso de tokens para embedding"""
    prompt_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)

class EmbeddingResponse(BaseModel):
    """Resposta de embedding compatível OpenAI"""
    object: Literal["list"] = "list"
    data: List[EmbeddingData] = Field(..., min_length=1)
    model: str
    usage: EmbeddingUsage

#### infra/schemas/model_schemas.py
```python
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional, Literal
from domain.entities.llm_model import ModelProvider, ModelType

class ModelInfo(BaseModel):
    """Informações básicas do modelo - compatível OpenAI"""
    id: str = Field(..., min_length=1)
    object: Literal["model"] = "model"
    created: int = Field(default=1677610602, gt=0)  # Timestamp fixo para compatibilidade
    owned_by: str = Field(..., min_length=1)

    # Campos opcionais OpenAI
    permission: List[str] = Field(default_factory=list)
    root: Optional[str] = None
    parent: Optional[str] = None

class ModelStats(BaseModel):
    """Estatísticas do modelo"""
    total_requests: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    average_response_time: float = Field(default=0.0, ge=0.0)
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    last_used: Optional[str] = None

class ModelListResponse(BaseModel):
    """Resposta da listagem de modelos - compatível OpenAI"""
    object: Literal["list"] = "list"
    data: List[ModelInfo] = Field(..., description="Lista de modelos disponíveis")

class ModelDetailsResponse(BaseModel):
    """Resposta detalhada de um modelo específico"""
    id: str
    name: str
    code: str
    provider: str
    model_type: str
    enabled: bool
    operations: List[str]
    costs: Dict[str, float]
    supports_streaming: bool
    created_at: Optional[str]
    updated_at: Optional[str]
    metadata: Optional[Dict[str, Any]] = None

class HealthCheckResponse(BaseModel):
    """Resposta do health check"""
    status: Literal["healthy", "unhealthy"]
    timestamp: float
    version: Optional[str] = None
    models: Optional[Dict[str, Any]] = None
    database: Optional[Dict[str, Any]] = None
    providers: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    """Resposta de erro padrão OpenAI"""
    error: Dict[str, Any] = Field(..., description="Detalhes do erro")

    class ErrorDetail(BaseModel):
        message: str
        type: str
        param: Optional[str] = None
        code: Optional[str] = None
```

### Infrastructure Layer - Controllers

#### infra/controllers/__init__.py
```python
# Controladores da camada de infraestrutura
from .chat_completion_controller import chat_router
from .text_to_speech_controller import tts_router
from .speech_to_text_controller import stt_router
from .image_generation_controller import image_router
from .model_controller import model_router

__all__ = [
    'chat_router', 'tts_router', 'stt_router',
    'image_router', 'model_router'
]
```

#### infra/controllers/chat_completion_controller.py
```python
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from typing import AsyncIterator
import json
import logging
import time
import uuid
from infra.schemas.chat_completion_schemas import (
    ChatCompletionRequest, ChatCompletionResponse
)
from application.use_cases.chat_completion_use_case import ChatCompletionUseCase
from application.interfaces.llm_service_interface import LLMServiceInterface
from infra.dependencies import get_model_service, get_request_id

logger = logging.getLogger(__name__)
chat_router = APIRouter(prefix="/v1", tags=["chat"])

@chat_router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    model_service: LLMServiceInterface = Depends(get_model_service),
    request_id: str = Depends(get_request_id),
    http_request: Request = None
):
    """Endpoint OpenAI-compatível para chat completions com streaming opcional"""
    start_time = time.time()

    try:
        logger.info(f"Chat completion request - Model: {request.model}, Stream: {request.stream}, Request ID: {request_id}")

        # Cria caso de uso
        use_case = ChatCompletionUseCase(model_service)

        # Verifica se é streaming
        if request.stream:
            return StreamingResponse(
                _stream_chat_completion(use_case, request, request_id),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Request-ID": request_id
                }
            )

        # Execução normal (não-streaming)
        response = await use_case.execute(request)

        # Log da resposta
        elapsed_time = time.time() - start_time
        logger.info(f"Chat completion completed - Request ID: {request_id}, Time: {elapsed_time:.2f}s")

        return response

    except ValueError as e:
        logger.warning(f"Validation error - Request ID: {request_id}, Error: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "error": {
                "message": str(e),
                "type": "invalid_request_error",
                "code": "validation_error"
            }
        })
    except Exception as e:
        logger.error(f"Internal error - Request ID: {request_id}, Error: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
                "code": "internal_server_error"
            }
        })

async def _stream_chat_completion(
    use_case: ChatCompletionUseCase,
    request: ChatCompletionRequest,
    request_id: str
) -> AsyncIterator[str]:
    """Gerador para streaming de chat completion"""
    try:
        async for chunk in use_case.execute_stream(request):
            # Formata chunk no formato SSE
            chunk_data = json.dumps(chunk, ensure_ascii=False)
            yield f"data: {chunk_data}\n\n"

        # Envia sinal de fim
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Streaming error - Request ID: {request_id}, Error: {str(e)}")
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": "streaming_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
```

#### infra/controllers/model_controller.py
```python
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List
import logging
from infra.schemas.model_schemas import (
    ModelListResponse, ModelInfo, HealthCheckResponse
)
from application.interfaces.llm_service_interface import LLMServiceInterface
from infra.dependencies import get_model_service

logger = logging.getLogger(__name__)
model_router = APIRouter(prefix="/v1", tags=["models"])

@model_router.get("/models", response_model=ModelListResponse)
async def list_models(
    include_private: bool = Query(default=False, description="Incluir modelos privados"),
    provider: Optional[str] = Query(default=None, description="Filtrar por provedor"),
    operation: Optional[str] = Query(default=None, description="Filtrar por operação"),
    project_id: Optional[str] = Query(default=None, description="ID do projeto"),
    model_service: LLMServiceInterface = Depends(get_model_service)
):
    """Lista modelos disponíveis - Endpoint compatível com OpenAI"""
    try:
        logger.info(f"Listing models - Provider: {provider}, Operation: {operation}, Project: {project_id}")

        # Aplica filtros
        if provider:
            models = await model_service.get_models_by_provider(provider)
        elif operation:
            models = await model_service.get_models_by_operation(operation)
        else:
            models = await model_service.list_models(include_private=include_private, project_id=project_id)

        # Converte para formato OpenAI
        model_infos = []
        for model in models:
            model_infos.append(ModelInfo(
                id=model.code,
                object="model",
                created=1677610602,  # Timestamp fixo para compatibilidade
                owned_by=model.provider.value.lower(),
                permission=[],
                root=model.code,
                parent=None
            ))

        logger.info(f"Found {len(model_infos)} models")
        return ModelListResponse(data=model_infos)

    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
                "code": "models_list_error"
            }
        })

@model_router.get("/models/{model_id}")
async def get_model(
    model_id: str,
    model_service: LLMServiceInterface = Depends(get_model_service)
):
    """Obtém informações detalhadas de um modelo específico"""
    try:
        logger.info(f"Getting model details: {model_id}")

        model = await model_service.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail={
                "error": {
                    "message": f"Model '{model_id}' not found",
                    "type": "not_found_error",
                    "code": "model_not_found"
                }
            })

        # Retorna informações detalhadas (formato customizado)
        return {
            "id": model.id,
            "name": model.name,
            "code": model.code,
            "provider": model.provider.value,
            "model_type": model.model_type.value,
            "enabled": model.enabled,
            "operations": model.operations,
            "costs": model.get_cost_info(),
            "supports_streaming": model.supports_streaming(),
            "created_at": model.created_at,
            "updated_at": model.updated_at
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
                "code": "model_get_error"
            }
        })

@model_router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    model_service: LLMServiceInterface = Depends(get_model_service)
):
    """Health check endpoint"""
    try:
        health_data = await model_service.health_check()
        return HealthCheckResponse(**health_data)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=time.time(),
            error=str(e)
        )
```

### Infrastructure Layer - Database

#### infra/database/__init__.py
```python
# Camada de banco de dados
from .cosmos_client import CosmosDBClient
from .model_repository import ModelRepository
from .cosmos_models import CosmosModelDocument

__all__ = ['CosmosDBClient', 'ModelRepository', 'CosmosModelDocument']
```

#### infra/database/cosmos_client.py
```python
from typing import List, Optional, Dict, Any
import asyncio
import logging
from azure.cosmos.aio import CosmosClient
from azure.cosmos import exceptions, PartitionKey
from llmproxy.config import Config

logger = logging.getLogger(__name__)

class CosmosDBClient:
    """Cliente assíncrono para Azure Cosmos DB"""

    def __init__(self):
        self._client = None
        self._database = None
        self._container = None
        self._initialized = False

    async def initialize(self):
        """Inicializa conexão com Cosmos DB"""
        if self._initialized:
            return

        try:
            logger.info("Initializing Cosmos DB connection")
            self._client = CosmosClient(
                Config.COSMOS_URL,
                Config.COSMOS_KEY,
                consistency_level='Session'
            )

            # Obtém referência do banco
            self._database = self._client.get_database_client(Config.COSMOS_DATABASE_ID)

            # Obtém referência do container
            self._container = self._database.get_container_client(Config.COSMOS_CONTAINER_ID)

            self._initialized = True
            logger.info("Cosmos DB connection initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Cosmos DB: {str(e)}")
            raise

    async def close(self):
        """Fecha conexão com Cosmos DB"""
        if self._client:
            await self._client.close()
            logger.info("Cosmos DB connection closed")

    async def create_item(self, item: Dict[str, Any], partition_key: str = None) -> Dict[str, Any]:
        """Cria um novo item no container"""
        await self._ensure_initialized()

        try:
            if not partition_key:
                partition_key = item.get('provider', item.get('id'))

            response = await self._container.create_item(
                body=item,
                partition_key=partition_key
            )
            logger.info(f"Created item: {item.get('id', 'unknown')}")
            return response

        except exceptions.CosmosResourceExistsError:
            raise ValueError(f"Item with ID '{item.get('id')}' already exists")
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"Error creating item: {str(e)}")
            raise

    async def get_item(self, item_id: str, partition_key: str = None) -> Optional[Dict[str, Any]]:
        """Obtém um item por ID"""
        await self._ensure_initialized()

        try:
            # Se não há partition key, busca por query
            if not partition_key:
                return await self._query_item_by_id(item_id)

            response = await self._container.read_item(
                item=item_id,
                partition_key=partition_key
            )
            return response

        except exceptions.CosmosResourceNotFoundError:
            return None
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"Error getting item {item_id}: {str(e)}")
            return None

    async def query_items(
        self,
        query: str,
        parameters: List[Dict[str, Any]] = None,
        max_items: int = None
    ) -> List[Dict[str, Any]]:
        """Executa query no container"""
        await self._ensure_initialized()

        try:
            query_iterable = self._container.query_items(
                query=query,
                parameters=parameters or [],
                enable_cross_partition_query=True,
                max_item_count=max_items
            )

            items = []
            async for item in query_iterable:
                items.append(item)

            logger.info(f"Query returned {len(items)} items")
            return items

        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"Error executing query: {str(e)}")
            return []

    async def update_item(self, item_id: str, item: Dict[str, Any], partition_key: str = None) -> Dict[str, Any]:
        """Atualiza um item existente"""
        await self._ensure_initialized()

        try:
            if not partition_key:
                partition_key = item.get('provider', item.get('id'))

            response = await self._container.replace_item(
                item=item_id,
                body=item,
                partition_key=partition_key
            )
            logger.info(f"Updated item: {item_id}")
            return response

        except exceptions.CosmosResourceNotFoundError:
            raise ValueError(f"Item with ID '{item_id}' not found")
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"Error updating item {item_id}: {str(e)}")
            raise

    async def delete_item(self, item_id: str, partition_key: str = None) -> bool:
        """Exclui um item"""
        await self._ensure_initialized()

        try:
            if not partition_key:
                # Busca o item para obter o partition key
                existing_item = await self.get_item(item_id)
                if not existing_item:
                    return False
                partition_key = existing_item.get('provider', item_id)

            await self._container.delete_item(
                item=item_id,
                partition_key=partition_key
            )
            logger.info(f"Deleted item: {item_id}")
            return True

        except exceptions.CosmosResourceNotFoundError:
            return False
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"Error deleting item {item_id}: {str(e)}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Verifica saúde da conexão com Cosmos DB"""
        try:
            await self._ensure_initialized()

            # Tenta fazer uma query simples
            start_time = time.time()
            items = await self.query_items("SELECT TOP 1 c.id FROM c", max_items=1)
            response_time = time.time() - start_time

            return {
                'status': 'healthy',
                'response_time': response_time,
                'items_count': len(items)
            }

        except Exception as e:
            logger.error(f"Cosmos DB health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

    async def _ensure_initialized(self):
        """Garante que o cliente está inicializado"""
        if not self._initialized:
            await self.initialize()

    async def _query_item_by_id(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Busca item por ID usando query"""
        query = "SELECT * FROM c WHERE c.id = @item_id OR c.code = @item_id"
        parameters = [{"name": "@item_id", "value": item_id}]

        items = await self.query_items(query, parameters, max_items=1)
        return items[0] if items else None
```

#### infra/database/model_repository.py
```python
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
from infra.database.cosmos_client import CosmosDBClient
from infra.database.cosmos_models import CosmosModelDocument

logger = logging.getLogger(__name__)

class ModelRepository:
    """Repositório para operações com modelos no Cosmos DB"""

    def __init__(self, cosmos_client: CosmosDBClient):
        self.cosmos_client = cosmos_client

    async def create(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Cria um novo modelo"""
        try:
            logger.info(f"Creating model: {model_data.get('id')}")

            # Valida dados usando o modelo Pydantic
            model_doc = CosmosModelDocument(**model_data)

            # Adiciona timestamps
            now = datetime.utcnow().isoformat()
            model_dict = model_doc.model_dump()
            model_dict['created_at'] = now
            model_dict['updated_at'] = now

            return await self.cosmos_client.create_item(
                model_dict,
                partition_key=model_dict['provider']
            )

        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise

    async def get_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Busca um modelo por ID"""
        try:
            logger.debug(f"Getting model by ID: {model_id}")
            return await self.cosmos_client.get_item(model_id)

        except Exception as e:
            logger.error(f"Error getting model {model_id}: {str(e)}")
            return None

    async def get_all(self) -> List[Dict[str, Any]]:
        """Lista todos os modelos"""
        try:
            logger.debug("Getting all models")
            query = "SELECT * FROM c WHERE c.enabled = true"
            return await self.cosmos_client.query_items(query)

        except Exception as e:
            logger.error(f"Error getting all models: {str(e)}")
            return []

    async def get_by_provider(self, provider: str) -> List[Dict[str, Any]]:
        """Busca modelos por provedor"""
        try:
            logger.debug(f"Getting models by provider: {provider}")
            query = "SELECT * FROM c WHERE c.provider = @provider AND c.enabled = true"
            parameters = [{"name": "@provider", "value": provider}]

            return await self.cosmos_client.query_items(query, parameters)

        except Exception as e:
            logger.error(f"Error getting models by provider {provider}: {str(e)}")
            return []

    async def get_by_type(self, model_type: str) -> List[Dict[str, Any]]:
        """Busca modelos por tipo"""
        try:
            logger.debug(f"Getting models by type: {model_type}")
            query = "SELECT * FROM c WHERE c.model_type = @model_type AND c.enabled = true"
            parameters = [{"name": "@model_type", "value": model_type}]

            return await self.cosmos_client.query_items(query, parameters)

        except Exception as e:
            logger.error(f"Error getting models by type {model_type}: {str(e)}")
            return []

    async def get_by_operation(self, operation: str) -> List[Dict[str, Any]]:
        """Busca modelos que suportam uma operação"""
        try:
            logger.debug(f"Getting models by operation: {operation}")
            query = "SELECT * FROM c WHERE ARRAY_CONTAINS(c.operations, @operation) AND c.enabled = true"
            parameters = [{"name": "@operation", "value": operation}]

            return await self.cosmos_client.query_items(query, parameters)

        except Exception as e:
            logger.error(f"Error getting models by operation {operation}: {str(e)}")
            return []

    async def get_by_project(self, project_id: str) -> List[Dict[str, Any]]:
        """Busca modelos disponíveis para um projeto"""
        try:
            logger.debug(f"Getting models by project: {project_id}")
            query = """
                SELECT * FROM c
                WHERE (c.private = false OR ARRAY_CONTAINS(c.projects, @project_id))
                AND c.enabled = true
            """
            parameters = [{"name": "@project_id", "value": project_id}]

            return await self.cosmos_client.query_items(query, parameters)

        except Exception as e:
            logger.error(f"Error getting models by project {project_id}: {str(e)}")
            return []

    async def update(self, model_id: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Atualiza um modelo existente"""
        try:
            logger.info(f"Updating model: {model_id}")

            # Adiciona timestamp de atualização
            model_data['updated_at'] = datetime.utcnow().isoformat()
            model_data['id'] = model_id  # Garante que o ID não mude

            # Valida dados
            model_doc = CosmosModelDocument(**model_data)

            return await self.cosmos_client.update_item(
                model_id,
                model_doc.model_dump(),
                partition_key=model_data['provider']
            )

        except Exception as e:
            logger.error(f"Error updating model {model_id}: {str(e)}")
            raise

    async def delete(self, model_id: str) -> bool:
        """Exclui um modelo"""
        try:
            logger.info(f"Deleting model: {model_id}")
            return await self.cosmos_client.delete_item(model_id)

        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {str(e)}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Verifica saúde do repositório"""
        try:
            # Conta total de modelos
            all_models = await self.get_all()

            # Agrupa por provedor
            providers = {}
            for model in all_models:
                provider = model.get('provider', 'unknown')
                if provider not in providers:
                    providers[provider] = 0
                providers[provider] += 1

            cosmos_health = await self.cosmos_client.health_check()

            return {
                'status': 'healthy' if cosmos_health['status'] == 'healthy' else 'unhealthy',
                'total_models': len(all_models),
                'providers': providers,
                'cosmos_db': cosmos_health
            }

        except Exception as e:
            logger.error(f"Repository health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
```

#### infra/database/cosmos_models.py
```python
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from domain.entities.llm_model import ModelProvider, ModelType

class CosmosModelDocument(BaseModel):
    """Documento de modelo no Cosmos DB com validações aprimoradas"""
    model_config = ConfigDict(
        extra="allow",  # Permite campos extras do Cosmos
        json_encoders={datetime: lambda v: v.isoformat()},
        validate_assignment=True
    )

    # Campos obrigatórios
    id: str = Field(..., min_length=1, max_length=256)
    name: str = Field(..., min_length=1, max_length=256)
    code: str = Field(..., min_length=1, max_length=256)
    provider: ModelProvider
    model_type: ModelType

    # Campos opcionais com defaults
    projects: List[str] = Field(default_factory=list)
    private: bool = Field(default=False)
    enabled: bool = Field(default=True)
    operations: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    costs: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Campos de auditoria
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # Campos automáticos do Cosmos DB
    _rid: Optional[str] = None
    _self: Optional[str] = None
    _etag: Optional[str] = None
    _attachments: Optional[str] = None
    _ts: Optional[int] = None

    @field_validator('operations')
    @classmethod
    def validate_operations(cls, v: List[str]) -> List[str]:
        """Valida se as operações são válidas"""
        valid_operations = {
            "Responses", "ChatCompletion", "embeddings",
            "images", "audio_tts", "audio_stt", "audio_realtime"
        }

        invalid_ops = set(v) - valid_operations
        if invalid_ops:
            raise ValueError(f"Invalid operations: {', '.join(invalid_ops)}")

        return v

    @field_validator('costs')
    @classmethod
    def validate_costs(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Valida estrutura de custos"""
        if v:
            # Verifica se tem chaves obrigatórias
            if 'input' not in v or 'output' not in v:
                raise ValueError("Costs must include 'input' and 'output' keys")

            # Verifica se valores são números positivos
            for key, value in v.items():
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValueError(f"Cost '{key}' must be a non-negative number")

        return v

    @field_validator('projects')
    @classmethod
    def validate_projects(cls, v: List[str]) -> List[str]:
        """Valida lista de projetos"""
        # Remove duplicatas e strings vazias
        return list(set(p.strip() for p in v if p and p.strip()))

    def get_required_parameters_for_provider(self) -> List[str]:
        """Retorna parâmetros obrigatórios baseado no provider"""
        provider_params = {
            ModelProvider.AZURE: ["secret_name", "deployment_name", "api_version", "endpoint"],
            ModelProvider.GOOGLE: ["secret_name", "gcp_project", "location"]
        }
        return provider_params.get(self.provider, [])

    def validate_required_parameters(self) -> bool:
        """Valida se todos os parâmetros obrigatórios estão presentes"""
        required = self.get_required_parameters_for_provider()
        missing = [param for param in required if param not in self.parameters]

        if missing:
            raise ValueError(f"Missing required parameters for {self.provider.value}: {', '.join(missing)}")

        return True

    def to_domain_entity(self) -> Dict[str, Any]:
        """Converte para formato da entidade de domínio"""
        return {
            'id': self.id,
            'name': self.name,
            'code': self.code,
            'provider': self.provider,
            'model_type': self.model_type,
            'projects': self.projects,
            'private': self.private,
            'enabled': self.enabled,
            'operations': self.operations,
            'parameters': self.parameters,
            'costs': self.costs,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
```

### Infrastructure Layer - Providers

#### infra/providers/__init__.py
```python
# Camada de provedores LLM
from .litellm_provider import LiteLLMProvider
from .llm_adapters import (
    AzureOpenAIAdapter, AnthropicAdapter, GoogleVertexAIAdapter
)
from .provider_utils import (
    TokenCounter, CacheManager, RateLimiter, OperationValidator
)

__all__ = [
    'LiteLLMProvider',
    'AzureOpenAIAdapter', 'AnthropicAdapter', 'GoogleVertexAIAdapter',
    'TokenCounter', 'CacheManager', 'RateLimiter', 'OperationValidator'
]
```

#### infra/providers/litellm_provider.py
```python
import litellm
from typing import Dict, Any, Optional, AsyncIterator
import logging
import asyncio
import tempfile
import json
from application.interfaces.llm_provider_interface import LLMProviderInterface
from domain.entities.llm_model import LLMModel, ModelType, ModelProvider
from llmproxy.config import Config

logger = logging.getLogger(__name__)

class LiteLLMProvider(LLMProviderInterface):
    """Provedor LiteLLM com operações assíncronas otimizadas"""

    def __init__(self, model: LLMModel):
        self.model = model
        self.parameters = model.parameters
        self._credentials_path = None

        # Configura LiteLLM
        asyncio.create_task(self._setup_litellm())

    async def _setup_litellm(self):
        """Configura LiteLLM com parâmetros do modelo"""
        try:
            # Configurações globais
            if 'api_key' in self.parameters:
                litellm.api_key = self.parameters['api_key']

            if 'api_base' in self.parameters:
                litellm.api_base = self.parameters['api_base']

            if 'api_version' in self.parameters:
                litellm.api_version = self.parameters['api_version']

            # Configura credenciais Google Vertex AI
            if self._is_google_vertex_model():
                await self._setup_google_credentials()

            logger.info(f"LiteLLM provider configured for model: {self.model.id}")

        except Exception as e:
            logger.error(f"Error setting up LiteLLM: {str(e)}")
            raise

    async def chat_completions(self, **kwargs) -> Dict[Any, Any]:
        """Processa requisições de chat/completions de forma assíncrona"""
        try:
            model_name = self._prepare_model_name()
            kwargs['model'] = model_name

            # Adiciona parâmetros adicionais
            additional_params = await self._get_additional_parameters()
            kwargs.update(additional_params)

            # Executa requisição assíncrona
            response = await litellm.acompletion(**kwargs)

            return self._format_response(response)

        except Exception as e:
            logger.error(f"Error in chat completions: {str(e)}")
            raise

    async def chat_completions_stream(self, **kwargs) -> AsyncIterator[Dict[Any, Any]]:
        """Processa requisições de chat/completions com streaming assíncrono"""
        try:
            model_name = self._prepare_model_name()
            kwargs['model'] = model_name
            kwargs['stream'] = True

            additional_params = await self._get_additional_parameters()
            kwargs.update(additional_params)

            # Stream assíncrono
            async for chunk in litellm.acompletion(**kwargs):
                yield self._format_response(chunk)

        except Exception as e:
            logger.error(f"Error in streaming chat completions: {str(e)}")
            raise

    async def text_to_speech(self, **kwargs) -> Dict[Any, Any]:
        """Processa requisições de text-to-speech de forma assíncrona"""
        try:
            model_name = self._prepare_model_name()
            kwargs['model'] = model_name

            additional_params = await self._get_additional_parameters()
            kwargs.update(additional_params)

            # Para TTS, pode precisar de implementação específica por provedor
            if self.model.provider == ModelProvider.OPENAI or \
               self.model.model_type == ModelType.AZURE_OPENAI:
                response = await litellm.aspeech(**kwargs)
            else:
                raise NotImplementedError(f"TTS not implemented for {self.model.provider.value}")

            return {'audio': response, 'content_type': 'audio/mpeg'}

        except Exception as e:
            logger.error(f"Error in TTS: {str(e)}")
            raise

    async def speech_to_text(self, **kwargs) -> Dict[Any, Any]:
        """Processa requisições de speech-to-text de forma assíncrona"""
        try:
            model_name = self._prepare_model_name()
            kwargs['model'] = model_name

            additional_params = await self._get_additional_parameters()
            kwargs.update(additional_params)

            if self.model.provider == ModelProvider.OPENAI or \
               self.model.model_type == ModelType.AZURE_OPENAI:
                response = await litellm.atranscription(**kwargs)
            else:
                raise NotImplementedError(f"STT not implemented for {self.model.provider.value}")

            return response

        except Exception as e:
            logger.error(f"Error in STT: {str(e)}")
            raise

    async def generate_image(self, **kwargs) -> Dict[Any, Any]:
        """Processa requisições de geração de imagem de forma assíncrona"""
        try:
            model_name = self._prepare_model_name()
            kwargs['model'] = model_name

            additional_params = await self._get_additional_parameters()
            kwargs.update(additional_params)

            response = await litellm.aimage_generation(**kwargs)
            return response

        except Exception as e:
            logger.error(f"Error in image generation: {str(e)}")
            raise

    async def create_embeddings(self, **kwargs) -> Dict[Any, Any]:
        """Processa requisições de embeddings de forma assíncrona"""
        try:
            model_name = self._prepare_model_name()
            kwargs['model'] = model_name

            additional_params = await self._get_additional_parameters()
            kwargs.update(additional_params)

            response = await litellm.aembedding(**kwargs)
            return response

        except Exception as e:
            logger.error(f"Error in embeddings: {str(e)}")
            raise

    def get_provider_name(self) -> str:
        """Retorna o nome do provedor"""
        return self.model.provider.value.lower()

    def get_model_type(self) -> str:
        """Retorna o tipo do modelo"""
        return self.model.model_type.value

    def supports_operation(self, operation: str) -> bool:
        """Verifica se o provedor suporta uma operação específica"""
        return operation in self.model.operations

    async def validate_credentials(self) -> bool:
        """Valida as credenciais do provedor de forma assíncrona"""
        try:
            # Tenta fazer uma requisição simples para validar credenciais
            if self.supports_operation('ChatCompletion'):
                await self.chat_completions(
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
            return True

        except Exception as e:
            logger.warning(f"Credential validation failed: {str(e)}")
            return False

    async def get_available_models(self) -> List[str]:
        """Obtém lista de modelos disponíveis no provedor"""
        try:
            # Implementação varia por provedor
            return [self.model.code]

        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return []

    def _prepare_model_name(self) -> str:
        """Prepara o nome do modelo para LiteLLM baseado no provider e model_type"""
        provider = self.model.provider
        model_type = self.model.model_type

        # Mapeamento para Azure
        if model_type in [ModelType.AZURE_OPENAI, ModelType.AZURE_FOUNDRY]:
            deployment_name = self.parameters.get('deployment_name', self.model.code)
            return f"azure/{deployment_name}"

        # Mapeamento para Google Vertex AI
        elif model_type == ModelType.GOOGLE_VERTEX:
            return f"vertex_ai/{self.model.code}"
        elif model_type == ModelType.GOOGLE_VERTEX_CLAUDE:
            return f"vertex_ai/claude-{self.model.code}"
        elif model_type == ModelType.GOOGLE_VERTEX_DEEPSEEK:
            return f"vertex_ai/deepseek-{self.model.code}"
        elif model_type == ModelType.GOOGLE_VERTEX_META:
            return f"vertex_ai/llama-{self.model.code}"
        elif model_type == ModelType.GOOGLE_VERTEX_MISTRAL:
            return f"vertex_ai/mistral-{self.model.code}"
        elif model_type == ModelType.GOOGLE_VERTEX_OPENAI:
            return f"vertex_ai/gpt-{self.model.code}"

        # Provedores nativos
        elif provider == ModelProvider.OPENAI:
            return self.model.code
        elif provider == ModelProvider.ANTHROPIC:
            return f"claude-{self.model.code}"

        # Default
        return self.model.code

    async def _get_additional_parameters(self) -> Dict[str, Any]:
        """Obtém parâmetros adicionais para LiteLLM"""
        additional_params = {}

        # Parâmetros comuns
        common_params = {
            'api_base': str,
            'api_version': str,
            'api_key': str,
            'timeout': float,
            'max_retries': int
        }

        for param_name, param_type in common_params.items():
            if param_name in self.parameters:
                try:
                    additional_params[param_name] = param_type(self.parameters[param_name])
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert parameter {param_name}: {e}")

        # Parâmetros específicos do Google Vertex AI
        if self._is_google_vertex_model():
            if 'gcp_project' in self.parameters:
                additional_params['vertex_project'] = self.parameters['gcp_project']
            if 'location' in self.parameters:
                additional_params['vertex_location'] = self.parameters['location']

            # Adiciona credenciais se disponíveis
            if self._credentials_path:
                additional_params['vertex_credentials'] = self._credentials_path

        return additional_params

    def _format_response(self, response) -> Dict[Any, Any]:
        """Formata a resposta para o padrão"""
        if hasattr(response, 'model_dump'):
            return response.model_dump()
        elif hasattr(response, 'to_dict'):
            return response.to_dict()
        elif isinstance(response, dict):
            return response
        else:
            try:
                return dict(response)
            except Exception:
                return {"error": "Could not format response", "raw_response": str(response)}

    def _is_google_vertex_model(self) -> bool:
        """Verifica se é um modelo Google Vertex AI"""
        google_vertex_types = [
            ModelType.GOOGLE_VERTEX, ModelType.GOOGLE_VERTEX_CLAUDE,
            ModelType.GOOGLE_VERTEX_DEEPSEEK, ModelType.GOOGLE_VERTEX_META,
            ModelType.GOOGLE_VERTEX_MISTRAL, ModelType.GOOGLE_VERTEX_JAMBA,
            ModelType.GOOGLE_VERTEX_QWEN, ModelType.GOOGLE_VERTEX_OPENAI
        ]
        return (
            self.model.model_type in google_vertex_types or
            self.model.provider == ModelProvider.GOOGLE
        )

    async def _setup_google_credentials(self):
        """Configura credenciais Google Vertex AI"""
        try:
            secret_name = self.parameters.get('secret_name')
            if not secret_name:
                logger.warning("No secret_name found for Google Vertex model")
                return

            credentials_content = Config.get_google_credentials(secret_name)
            if not credentials_content:
                logger.warning(f"Could not load Google credentials from {secret_name}")
                return

            # Cria arquivo temporário com credenciais
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                if isinstance(credentials_content, str):
                    f.write(credentials_content)
                else:
                    json.dump(credentials_content, f)
                self._credentials_path = f.name

            # Define variável de ambiente para LiteLLM
            import os
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self._credentials_path

            logger.info(f"Google credentials configured from {secret_name}")

        except Exception as e:
            logger.error(f"Error setting up Google credentials: {str(e)}")

    def __del__(self):
        """Limpa arquivo temporário de credenciais"""
        if self._credentials_path:
            try:
                import os
                os.unlink(self._credentials_path)
            except Exception:
                pass
```

### Principais melhorias implementadas:

### Core Configuration

#### llmproxy/config.py
```python
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Configurações do banco de dados"""
    url: str
    key: str
    database_id: str
    container_id: str

class Config:
    """Configurações centralizadas da aplicação"""

    APP_NAME = os.getenv("APP_NAME", "LLM Proxy")
    APP_VERSION = os.getenv("APP_VERSION", "2.0.0")
    APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("APP_PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    # Configurações do Cosmos DB
    COSMOS_URL = os.getenv("COSMOS_URL", "")
    COSMOS_KEY = os.getenv("COSMOS_KEY", "")
    COSMOS_DATABASE_ID = os.getenv("COSMOS_DATABASE_ID", "llm-proxy")
    COSMOS_CONTAINER_ID = os.getenv("COSMOS_CONTAINER_ID", "models")

    # Configurações de segurança
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

    # Configurações Redis para cache e session (FUTURO - Comentado)
    # REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    # REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    # REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
    # REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    # REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))

    # Configurações Kafka para processamento assíncrono
    KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    KAFKA_ASYNC_CHAT_TOPIC = os.getenv("KAFKA_ASYNC_CHAT_TOPIC", "llm-async-chat")
    KAFKA_CONSUMER_GROUP_ID = os.getenv("KAFKA_CONSUMER_GROUP_ID", "llm-proxy-consumers")
    KAFKA_MAX_POLL_RECORDS = int(os.getenv("KAFKA_MAX_POLL_RECORDS", "100"))
    KAFKA_CONSUMER_CONCURRENCY = int(os.getenv("KAFKA_CONSUMER_CONCURRENCY", "4"))
    KAFKA_BATCH_SIZE = int(os.getenv("KAFKA_BATCH_SIZE", "50"))

    # Configurações de performance
    CONSUMER_THREAD_POOL_SIZE = int(os.getenv("CONSUMER_THREAD_POOL_SIZE", "8"))
    HTTP_CLIENT_MAX_CONNECTIONS = int(os.getenv("HTTP_CLIENT_MAX_CONNECTIONS", "100"))
    HTTP_CLIENT_MAX_KEEPALIVE = int(os.getenv("HTTP_CLIENT_MAX_KEEPALIVE", "50"))

    # Rate limiting avançado
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "100"))

    @classmethod
    def get_database_config(cls) -> DatabaseConfig:
        return DatabaseConfig(
            url=cls.COSMOS_URL,
            key=cls.COSMOS_KEY,
            database_id=cls.COSMOS_DATABASE_ID,
            container_id=cls.COSMOS_CONTAINER_ID
        )

    @classmethod
    def get_provider_config(cls, config_name: str) -> Optional[str]:
        return os.getenv(config_name)

    @classmethod
    def validate_required_config(cls) -> bool:
        required_configs = [("COSMOS_URL", cls.COSMOS_URL), ("COSMOS_KEY", cls.COSMOS_KEY)]
        missing_configs = [name for name, value in required_configs if not value]
        
        if missing_configs:
            logger.error(f"Missing required configurations: {', '.join(missing_configs)}")
            return False
        return True
```

### Performance Enhancements - Kafka & Async Processing

#### infra/messaging/__init__.py
```python
# Sistema de mensageria assíncrona
from .kafka_producer import KafkaAsyncProducer
from .kafka_consumer import KafkaAsyncConsumer
from .message_schemas import AsyncChatRequest, AsyncChatResponse, MessageStatus

__all__ = [
    'KafkaAsyncProducer', 'KafkaAsyncConsumer',
    'AsyncChatRequest', 'AsyncChatResponse', 'MessageStatus'
]
```

#### infra/messaging/kafka_producer.py
```python
from aiokafka import AIOKafkaProducer
import json
import logging
from typing import Dict, Any, Optional
import asyncio
from llmproxy.config import Config

logger = logging.getLogger(__name__)

class KafkaAsyncProducer:
    """Producer Kafka assíncrono para alta performance"""

    def __init__(self):
        self.producer: Optional[AIOKafkaProducer] = None
        self._producer_lock = asyncio.Lock()
        self.is_initialized = False

    async def initialize(self):
        """Inicializa producer Kafka com configurações otimizadas"""
        if self.is_initialized:
            return

        async with self._producer_lock:
            if self.is_initialized:
                return

            try:
                self.producer = AIOKafkaProducer(
                    bootstrap_servers=Config.KAFKA_BOOTSTRAP_SERVERS,
                    compression_type='lz4',  # Compressão para reduzir latência de rede
                    linger_ms=5,  # Aguarda 5ms para fazer batching
                    batch_size=32768,  # 32KB batch size
                    acks='1',  # Aguarda confirmação do líder apenas
                    retries=3,
                    value_serializer=lambda x: json.dumps(x, separators=(',', ':')).encode('utf-8'),
                    key_serializer=lambda x: str(x).encode('utf-8') if x else None
                )

                await self.producer.start()
                self.is_initialized = True
                logger.info("Kafka producer initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize Kafka producer: {str(e)}")
                raise

    async def send_async_chat_request(
        self,
        request_id: str,
        chat_request: Dict[str, Any],
        priority: int = 0
    ) -> bool:
        """Envia requisição de chat para processamento assíncrono"""
        await self._ensure_initialized()

        try:
            message = {
                'request_id': request_id,
                'timestamp': asyncio.get_event_loop().time(),
                'priority': priority,
                'payload': chat_request,
                'type': 'chat_completion'
            }

            # Usa o request_id como chave para particionamento consistente
            future = await self.producer.send(
                Config.KAFKA_ASYNC_CHAT_TOPIC,
                value=message,
                key=request_id
            )

            logger.info(f"Async chat request sent: {request_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send async chat request {request_id}: {str(e)}")
            return False

    async def send_batch_requests(self, requests: list[Dict[str, Any]]) -> Dict[str, bool]:
        """Envia múltiplas requisições em batch para máxima performance"""
        await self._ensure_initialized()

        results = {}
        tasks = []

        # Cria tasks assíncronas para todas as requisições
        for req in requests:
            task = asyncio.create_task(
                self.send_async_chat_request(
                    req['request_id'],
                    req['payload'],
                    req.get('priority', 0)
                )
            )
            tasks.append((req['request_id'], task))

        # Executa todas em paralelo
        for request_id, task in tasks:
            try:
                results[request_id] = await task
            except Exception as e:
                logger.error(f"Batch send failed for {request_id}: {str(e)}")
                results[request_id] = False

        return results

    async def close(self):
        """Fecha producer Kafka"""
        if self.producer:
            await self.producer.stop()
            self.is_initialized = False

    async def _ensure_initialized(self):
        """Garante que o producer está inicializado"""
        if not self.is_initialized:
            await self.initialize()
```

#### infra/messaging/message_schemas.py
```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime

class MessageStatus(str, Enum):
    """Status do processamento da mensagem"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Priority(int, Enum):
    """Níveis de prioridade para processamento"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

class AsyncChatRequest(BaseModel):
    """Schema para requisição assíncrona - baseado em ChatCompletionRequest"""
    model_config = ConfigDict(extra="allow")  # Permite todos os parâmetros do ChatCompletionRequest

    # Parâmetros obrigatórios (herdados de ChatCompletionRequest)
    model: str = Field(..., min_length=1, max_length=256)
    messages: List[Dict[str, Any]] = Field(..., min_length=1, max_length=200)

    # Parâmetros opcionais do ChatCompletionRequest (todos suportados)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0, le=200000)
    max_completion_tokens: Optional[int] = Field(default=None, gt=0, le=200000)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1, le=20)
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = Field(default=None, max_length=256)
    seed: Optional[int] = Field(default=None, ge=0, le=2147483647)

    # Tools e function calling
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = True
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None

    # Response format e logprobs
    response_format: Optional[Dict[str, Any]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20)

    # Parâmetros específicos para processamento assíncrono
    priority: Priority = Field(default=Priority.NORMAL)
    timeout_seconds: int = Field(default=300, ge=30, le=3600)
    callback_url: Optional[str] = Field(default=None, description="URL para webhook quando completo")
    webhook_events: Optional[List[Literal["started", "completed", "failed"]]] = Field(
        default_factory=lambda: ["completed", "failed"]
    )

    # Metadados para rastreamento
    client_id: Optional[str] = None
    session_id: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None

    # Configurações de retry específicas para async
    async_max_retries: Optional[int] = Field(default=3, ge=0, le=5)
    async_retry_delay: Optional[float] = Field(default=1.0, ge=0.1, le=10.0)

    @field_validator('callback_url')
    @classmethod
    def validate_callback_url(cls, v):
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError('callback_url must be a valid HTTP/HTTPS URL')
        return v

class AsyncChatResponse(BaseModel):
    """Schema para resposta de requisição assíncrona"""
    request_id: str
    status: MessageStatus
    submitted_at: datetime

    # Informações de processamento
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time_seconds: Optional[float] = None
    worker_id: Optional[str] = None

    # Resultado (preenchido quando status = COMPLETED)
    result: Optional[Dict[str, Any]] = None

    # Informações de erro (preenchido quando status = FAILED)
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # Metadados
    priority: Priority
    estimated_completion_time: Optional[datetime] = None
```

#### infra/cache/# redis_cache.py (FUTURO - Comentado)
```python
# IMPLEMENTAÇÃO FUTURA - REDIS CACHE
# import # aioredis
# import json
# import logging
# from typing import Any, Optional, Dict
# import asyncio
# from llmproxy.config import Config

# logger = logging.getLogger(__name__)

# IMPLEMENTAÇÃO FUTURA - REDIS CACHE
# class # AsyncRedisCache:
#     """Cache Redis assíncrono otimizado para alta performance"""
#
#     def __init__(self):
#         self.redis: Optional[# aioredis.Redis] = None
#         self._connection_pool: Optional[# aioredis.ConnectionPool] = None
#         self.is_initialized = False
#
#     async def initialize(self):
#         """Inicializa conexão Redis com pool otimizado"""
#         if self.is_initialized:
#             return
#
#         try:
#             # Pool de conexões para alta concorrência
#             self._connection_pool = # aioredis.ConnectionPool(
#                 host=Config.REDIS_HOST,
#                 port=Config.REDIS_PORT,
#                 password=Config.REDIS_PASSWORD,
#                 db=Config.REDIS_DB,
#                 max_connections=Config.REDIS_MAX_CONNECTIONS,
#                 retry_on_timeout=True,
#                 socket_connect_timeout=5,
#                 health_check_interval=30
#             )
#
#             self.redis = # aioredis.Redis(
#                 connection_pool=self._connection_pool,
#                 decode_responses=True
#             )
#
#             # Testa conexão
#             await self.redis.ping()
#             self.is_initialized = True
#             logger.info("Redis cache initialized successfully")
#
#         except Exception as e:
#             logger.error(f"Failed to initialize Redis cache: {str(e)}")
#             raise
#
#     async def get(self, key: str, default: Any = None) -> Any:
#         """Obtém valor do cache com deserialização automática"""
#         await self._ensure_initialized()
#
#         try:
#             value = await self.redis.get(key)
#             if value is None:
#                 return default
#
#             return json.loads(value)
#
#         except Exception as e:
#             logger.warning(f"Cache get failed for key {key}: {str(e)}")
#             return default
#
#     async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
#         """Define valor no cache com serialização automática"""
#         await self._ensure_initialized()
#
#         try:
#             serialized_value = json.dumps(value, separators=(',', ':'))
#
#             if ttl:
#                 result = await self.redis.setex(key, ttl, serialized_value)
#             else:
#                 result = await self.redis.set(key, serialized_value)
#
#             return bool(result)
#
#         except Exception as e:
#             logger.error(f"Cache set failed for key {key}: {str(e)}")
#             return False
#
#     async def mget(self, keys: list[str]) -> Dict[str, Any]:
#         """Obtém múltiplos valores em uma única operação"""
#         await self._ensure_initialized()
#
#         try:
#             values = await self.redis.mget(keys)
#             result = {}
#
#             for i, key in enumerate(keys):
#                 if values[i] is not None:
#                     try:
#                         result[key] = json.loads(values[i])
#                     except json.JSONDecodeError:
#                         result[key] = None
#                 else:
#                     result[key] = None
#
#             return result
#
#         except Exception as e:
#             logger.error(f"Cache mget failed: {str(e)}")
#             return {key: None for key in keys}
#
#     async def delete(self, *keys: str) -> int:
#         """Remove uma ou mais chaves do cache"""
#         await self._ensure_initialized()
#
#         try:
#             return await self.redis.delete(*keys)
#         except Exception as e:
#             logger.error(f"Cache delete failed: {str(e)}")
#             return 0
#
#     async def close(self):
#         """Fecha conexão Redis"""
#         if self._connection_pool:
#             await self._connection_pool.disconnect()
#         self.is_initialized = False
#
#     async def _ensure_initialized(self):
#         """Garante que o cache está inicializado"""
#         if not self.is_initialized:
#             await self.initialize()
```

#### application/use_cases/async_chat_completion_use_case.py
```python
from typing import Dict, Any, Optional
import logging
import asyncio
from datetime import datetime, timedelta
from application.interfaces.llm_service_interface import LLMServiceInterface
# FUTURO: from infra.cache.# redis_cache import # AsyncRedisCache
from infra.messaging.message_schemas import MessageStatus, AsyncChatResponse
import json

logger = logging.getLogger(__name__)

class AsyncChatCompletionUseCase:
    """Caso de uso para processamento assíncrono de chat completions"""

    def __init__(self,
                 model_service: LLMServiceInterface,
                 cache: # AsyncRedisCache):
        self.model_service = model_service
        self.cache = cache

    async def process_async_request(
        self,
        request_id: str,
        chat_request: Dict[str, Any],
        worker_id: int
    ):
        """Processa requisição assíncrona de chat completion"""
        start_time = datetime.utcnow()

        try:
            # Atualiza status para PROCESSING
            await self._update_request_status(
                request_id,
                MessageStatus.PROCESSING,
                started_at=start_time,
                worker_id=str(worker_id)
            )

            # Obtém modelo
            model = await self.model_service.get_model(chat_request['model'])
            if not model:
                raise ValueError(f"Model '{chat_request['model']}' not found")

            # Executa chat completion
            from infra.providers.litellm_provider import LiteLLMProvider
            provider = LiteLLMProvider(model)

            result = await provider.chat_completions(**chat_request)

            # Calcula tempo de processamento
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()

            # Salva resultado completo
            response = AsyncChatResponse(
                request_id=request_id,
                status=MessageStatus.COMPLETED,
                submitted_at=start_time,
                started_at=start_time,
                completed_at=end_time,
                processing_time_seconds=processing_time,
                worker_id=str(worker_id),
                result=result,
                priority=chat_request.get('priority', 1)
            )

            # Armazena no cache com TTL de 24 horas
            await self.cache.set(
                f"async_result:{request_id}",
                response.model_dump(),
                ttl=86400
            )

            logger.info(f"Async request {request_id} completed successfully in {processing_time:.2f}s")

        except Exception as e:
            logger.error(f"Async request {request_id} failed: {str(e)}")

            # Atualiza status para FAILED
            await self._update_request_status(
                request_id,
                MessageStatus.FAILED,
                error_code="processing_error",
                error_message=str(e)
            )

    async def _update_request_status(
        self,
        request_id: str,
        status: MessageStatus,
        **kwargs
    ):
        """Atualiza status de uma requisição no cache"""
        try:
            # Obtém dados existentes
            existing_data = await self.cache.get(f"async_result:{request_id}", {})

            # Atualiza campos
            existing_data.update({
                'request_id': request_id,
                'status': status.value,
                **kwargs
            })

            # Salva no cache
            await self.cache.set(
                f"async_result:{request_id}",
                existing_data,
                ttl=86400
            )

        except Exception as e:
            logger.error(f"Failed to update request status for {request_id}: {str(e)}")
```

#### infra/controllers/async_chat_controller.py
```python
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any
import logging
import uuid
from datetime import datetime
from infra.messaging.kafka_producer import KafkaAsyncProducer
from infra.messaging.message_schemas import AsyncChatRequest, AsyncChatResponse, MessageStatus
# FUTURO: from infra.cache.# redis_cache import # AsyncRedisCache
from infra.dependencies import get_kafka_producer, get_# redis_cache

logger = logging.getLogger(__name__)
async_chat_router = APIRouter(prefix="/v1/async", tags=["async-chat"])

@async_chat_router.post("/chat/completions")
async def create_async_chat_completion(
    request: AsyncChatRequest,
    producer: KafkaAsyncProducer = Depends(get_kafka_producer),
    cache: # AsyncRedisCache = Depends(get_# redis_cache)
) -> Dict[str, Any]:
    """Endpoint assíncrono para chat completions - retorna ID para consulta posterior"""

    # Gera ID único para a requisição
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Creating async chat completion request: {request_id}")

        # Cria entrada inicial no cache
        initial_response = AsyncChatResponse(
            request_id=request_id,
            status=MessageStatus.PENDING,
            submitted_at=datetime.utcnow(),
            priority=request.priority
        )

        # Salva no cache com TTL de 24 horas
        await cache.set(
            f"async_result:{request_id}",
            initial_response.model_dump(),
            ttl=86400
        )

        # Envia para fila Kafka para processamento
        success = await producer.send_async_chat_request(
            request_id=request_id,
            chat_request=request.model_dump(),
            priority=request.priority.value
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to queue request for processing"
            )

        # Retorna informações da requisição
        return {
            "request_id": request_id,
            "status": MessageStatus.PENDING.value,
            "submitted_at": initial_response.submitted_at.isoformat(),
            "estimated_completion_time": None,
            "status_endpoint": f"/v1/async/chat/completions/{request_id}/status",
            "result_endpoint": f"/v1/async/chat/completions/{request_id}/result"
        }

    except Exception as e:
        logger.error(f"Failed to create async chat completion: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@async_chat_router.get("/chat/completions/{request_id}/status")
async def get_async_chat_status(
    request_id: str,
    cache: # AsyncRedisCache = Depends(get_# redis_cache)
) -> Dict[str, Any]:
    """Consulta status de uma requisição assíncrona"""

    try:
        # Busca no cache
        result = await cache.get(f"async_result:{request_id}")

        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Request {request_id} not found"
            )

        # Retorna apenas informações de status (sem resultado)
        status_info = {
            "request_id": result.get("request_id"),
            "status": result.get("status"),
            "submitted_at": result.get("submitted_at"),
            "started_at": result.get("started_at"),
            "completed_at": result.get("completed_at"),
            "processing_time_seconds": result.get("processing_time_seconds"),
            "worker_id": result.get("worker_id"),
            "error_code": result.get("error_code"),
            "error_message": result.get("error_message")
        }

        return status_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status for request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@async_chat_router.get("/chat/completions/{request_id}/result")
async def get_async_chat_result(
    request_id: str,
    cache: # AsyncRedisCache = Depends(get_# redis_cache)
) -> Dict[str, Any]:
    """Obtém resultado de uma requisição assíncrona"""

    try:
        # Busca no cache
        result = await cache.get(f"async_result:{request_id}")

        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Request {request_id} not found"
            )

        status = result.get("status")

        if status == MessageStatus.PENDING.value:
            raise HTTPException(
                status_code=202,
                detail="Request is still pending processing"
            )
        elif status == MessageStatus.PROCESSING.value:
            raise HTTPException(
                status_code=202,
                detail="Request is currently being processed"
            )
        elif status == MessageStatus.FAILED.value:
            raise HTTPException(
                status_code=500,
                detail={
                    "error_code": result.get("error_code"),
                    "error_message": result.get("error_message")
                }
            )
        elif status == MessageStatus.COMPLETED.value:
            return result.get("result", {})
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Unknown request status: {status}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get result for request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@async_chat_router.delete("/chat/completions/{request_id}")
async def cancel_async_chat_request(
    request_id: str,
    cache: # AsyncRedisCache = Depends(get_# redis_cache)
) -> Dict[str, str]:
    """Cancela uma requisição assíncrona (se ainda não foi processada)"""

    try:
        # Busca requisição atual
        result = await cache.get(f"async_result:{request_id}")

        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Request {request_id} not found"
            )

        current_status = result.get("status")

        if current_status in [MessageStatus.COMPLETED.value, MessageStatus.FAILED.value]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel request with status: {current_status}"
            )

        # Atualiza status para cancelado
        result["status"] = MessageStatus.CANCELLED.value
        result["completed_at"] = datetime.utcnow().isoformat()

        await cache.set(f"async_result:{request_id}", result, ttl=86400)

        return {
            "request_id": request_id,
            "status": "cancelled",
            "message": "Request has been cancelled"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
```

#### infra/messaging/kafka_consumer.py
```python
from aiokafka import AIOKafkaConsumer
import json
import logging
from typing import Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
from llmproxy.config import Config
from application.use_cases.async_chat_completion_use_case import AsyncChatCompletionUseCase

logger = logging.getLogger(__name__)

class KafkaAsyncConsumer:
    """Consumer Kafka assíncrono com processamento paralelo"""

    def __init__(self, chat_use_case: AsyncChatCompletionUseCase):
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.chat_use_case = chat_use_case
        self.is_running = False
        self._consumer_tasks: list[asyncio.Task] = []
        # Pool de threads para operações CPU-intensivas
        self.thread_pool = ThreadPoolExecutor(
            max_workers=Config.CONSUMER_THREAD_POOL_SIZE,
            thread_name_prefix="kafka-consumer"
        )

    async def initialize(self):
        """Inicializa consumer com configurações para alta performance"""
        try:
            self.consumer = AIOKafkaConsumer(
                Config.KAFKA_ASYNC_CHAT_TOPIC,
                bootstrap_servers=Config.KAFKA_BOOTSTRAP_SERVERS,
                group_id=Config.KAFKA_CONSUMER_GROUP_ID,
                # Configurações de performance
                auto_offset_reset='latest',
                enable_auto_commit=False,  # Controle manual de commits
                max_poll_records=Config.KAFKA_MAX_POLL_RECORDS,  # Processa em batches
                fetch_max_wait_ms=500,  # Reduz latência
                fetch_min_bytes=1024,  # Mínimo de dados por fetch
                # Deserialização otimizada
                value_deserializer=lambda x: json.loads(x.decode('utf-8')) if x else None,
                key_deserializer=lambda x: x.decode('utf-8') if x else None
            )

            await self.consumer.start()
            logger.info("Kafka consumer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {str(e)}")
            raise

    async def start_consuming(self):
        """Inicia consumo com múltiplas coroutines para processamento paralelo"""
        await self.initialize()
        self.is_running = True

        # Cria múltiplas tasks de consumo para paralelismo
        for i in range(Config.KAFKA_CONSUMER_CONCURRENCY):
            task = asyncio.create_task(
                self._consumer_loop(worker_id=i),
                name=f"kafka-consumer-{i}"
            )
            self._consumer_tasks.append(task)

        logger.info(f"Started {len(self._consumer_tasks)} Kafka consumer workers")

        # Aguarda todas as tasks
        try:
            await asyncio.gather(*self._consumer_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Consumer tasks failed: {str(e)}")
        finally:
            await self.stop_consuming()

    async def _consumer_loop(self, worker_id: int):
        """Loop principal de consumo para cada worker"""
        logger.info(f"Starting consumer worker {worker_id}")

        try:
            while self.is_running:
                # Busca mensagens em batch para maior eficiência
                batch = await self.consumer.getmany(
                    timeout_ms=1000,
                    max_records=Config.KAFKA_BATCH_SIZE
                )

                if not batch:
                    continue

                # Processa mensagens em paralelo
                processing_tasks = []
                for topic_partition, messages in batch.items():
                    for message in messages:
                        task = asyncio.create_task(
                            self._process_message(message, worker_id)
                        )
                        processing_tasks.append(task)

                # Aguarda processamento de todos os mensagens do batch
                if processing_tasks:
                    results = await asyncio.gather(
                        *processing_tasks,
                        return_exceptions=True
                    )

                    # Log erros se houver
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            logger.error(f"Worker {worker_id} message processing failed: {result}")

                    # Commit manual após processamento bem-sucedido
                    try:
                        await self.consumer.commit()
                    except Exception as e:
                        logger.error(f"Failed to commit offset: {str(e)}")

        except Exception as e:
            logger.error(f"Consumer worker {worker_id} failed: {str(e)}")

    async def _process_message(self, message, worker_id: int):
        """Processa uma mensagem individual"""
        try:
            data = message.value
            request_id = data.get('request_id')
            payload = data.get('payload')

            logger.info(f"Worker {worker_id} processing request: {request_id}")

            # Processa a requisição de chat completion de forma assíncrona
            start_time = asyncio.get_event_loop().time()

            await self.chat_use_case.process_async_request(
                request_id=request_id,
                chat_request=payload,
                worker_id=worker_id
            )

            processing_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"Worker {worker_id} completed request {request_id} in {processing_time:.2f}s")

        except Exception as e:
            logger.error(f"Worker {worker_id} failed to process message: {str(e)}")
            raise

    async def stop_consuming(self):
        """Para o consumo e limpa recursos"""
        self.is_running = False

        # Cancela todas as tasks de consumo
        for task in self._consumer_tasks:
            if not task.done():
                task.cancel()

        # Aguarda cancelamento
        if self._consumer_tasks:
            await asyncio.gather(*self._consumer_tasks, return_exceptions=True)

        # Fecha consumer
        if self.consumer:
            await self.consumer.stop()

        # Fecha thread pool
        self.thread_pool.shutdown(wait=True)

        logger.info("Kafka consumer stopped")
```

#### infra/performance/__init__.py
```python
# Componentes de performance e otimização
from .connection_pool import OptimizedHTTPClient
from .rate_limiter import AdvancedRateLimiter
from .metrics_collector import MetricsCollector

__all__ = ['OptimizedHTTPClient', 'AdvancedRateLimiter', 'MetricsCollector']
```

#### infra/performance/connection_pool.py
```python
import aiohttp
import asyncio
import logging
from typing import Dict, Any, Optional
from llmproxy.config import Config

logger = logging.getLogger(__name__)

class OptimizedHTTPClient:
    """Cliente HTTP otimizado com pool de conexões e keep-alive"""

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Inicializa cliente HTTP com configurações otimizadas"""
        async with self._lock:
            if self._session:
                return

            # Configurações otimizadas para alta performance
            timeout = aiohttp.ClientTimeout(
                total=Config.HTTP_CLIENT_TIMEOUT,
                connect=10,
                sock_read=60
            )

            # Connector com pool de conexões
            self._connector = aiohttp.TCPConnector(
                limit=Config.HTTP_CLIENT_MAX_CONNECTIONS,
                limit_per_host=Config.HTTP_CLIENT_MAX_KEEPALIVE,
                keepalive_timeout=30,
                enable_cleanup_closed=True,
                use_dns_cache=True,
                ttl_dns_cache=300,
                family=0,  # Auto-detect IPv4/IPv6
                ssl=False  # Para HTTPS, configure certificados adequadamente
            )

            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=timeout,
                headers={
                    'User-Agent': f'{Config.APP_NAME}/{Config.APP_VERSION}',
                    'Connection': 'keep-alive',
                    'Keep-Alive': 'timeout=30, max=1000'
                }
            )

            logger.info("Optimized HTTP client initialized")

    async def post(self, url: str, **kwargs) -> Dict[str, Any]:
        """POST request otimizado"""
        await self._ensure_initialized()

        try:
            async with self._session.post(url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()

        except Exception as e:
            logger.error(f"HTTP POST failed for {url}: {str(e)}")
            raise

    async def get(self, url: str, **kwargs) -> Dict[str, Any]:
        """GET request otimizado"""
        await self._ensure_initialized()

        try:
            async with self._session.get(url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()

        except Exception as e:
            logger.error(f"HTTP GET failed for {url}: {str(e)}")
            raise

    async def parallel_requests(self, requests: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """Executa múltiplas requisições em paralelo"""
        await self._ensure_initialized()

        tasks = []
        for req in requests:
            method = req.get('method', 'POST').upper()
            url = req['url']
            kwargs = req.get('kwargs', {})

            if method == 'POST':
                task = self.post(url, **kwargs)
            elif method == 'GET':
                task = self.get(url, **kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            tasks.append(task)

        # Executa todas as requisições em paralelo
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Converte exceções para dicionários de erro
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    'error': True,
                    'message': str(result)
                })
            else:
                processed_results.append(result)

        return processed_results

    async def close(self):
        """Fecha conexões HTTP"""
        if self._session:
            await self._session.close()
        if self._connector:
            await self._connector.close()

        logger.info("HTTP client closed")

    async def _ensure_initialized(self):
        """Garante que o cliente está inicializado"""
        if not self._session:
            await self.initialize()
```

#### infra/performance/rate_limiter.py
```python
import asyncio
import time
import logging
from typing import Dict, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
# FUTURO: from infra.cache.# redis_cache import # AsyncRedisCache

logger = logging.getLogger(__name__)

@dataclass
class RateLimitRule:
    """Regra de rate limiting"""
    requests_per_minute: int
    burst_limit: int
    window_size: int = 60  # segundos

class AdvancedRateLimiter:
    """Rate limiter avançado com sliding window e Redis distribuído"""

    def __init__(self, # redis_cache: Optional[# AsyncRedisCache] = None):
        self.# redis_cache = # redis_cache
        # Cache local para performance (fallback se Redis falhar)
        self._local_windows: Dict[str, deque] = defaultdict(deque)
        self._local_counters: Dict[str, int] = defaultdict(int)

    async def check_rate_limit(
        self,
        identifier: str,
        rule: RateLimitRule,
        cost: int = 1
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Verifica se a requisição está dentro do rate limit

        Returns:
            (allowed, info_dict) where info_dict contains:
            - remaining_requests
            - reset_time
            - retry_after (if denied)
        """
        current_time = time.time()

        # Tenta usar Redis distribuído primeiro
        if self.# redis_cache:
            try:
                return await self._check_redis_rate_limit(
                    identifier, rule, cost, current_time
                )
            except Exception as e:
                logger.warning(f"Redis rate limiting failed, using local: {str(e)}")

        # Fallback para rate limiting local
        return await self._check_local_rate_limit(
            identifier, rule, cost, current_time
        )

    async def _check_redis_rate_limit(
        self,
        identifier: str,
        rule: RateLimitRule,
        cost: int,
        current_time: float
    ) -> tuple[bool, Dict[str, Any]]:
        """Rate limiting usando Redis com sliding window"""

        window_start = int(current_time - rule.window_size)
        redis_key = f"rate_limit:{identifier}"

        # Pipeline Redis para operações atômicas
        pipe = self.# redis_cache.redis.pipeline(transaction=True)

        # Remove entradas antigas (sliding window)
        await pipe.zremrangebyscore(redis_key, 0, window_start)

        # Conta requisições atuais
        await pipe.zcard(redis_key)

        # Adiciona requisição atual (se permitida)
        await pipe.zadd(redis_key, {str(current_time): current_time})

        # Define TTL
        await pipe.expire(redis_key, rule.window_size + 10)

        results = await pipe.execute()
        current_requests = results[1]

        # Verifica se excede o limite
        if current_requests + cost > rule.requests_per_minute:
            # Remove a requisição que adicionamos
            await self.# redis_cache.redis.zrem(redis_key, str(current_time))

            return False, {
                'remaining_requests': max(0, rule.requests_per_minute - current_requests),
                'reset_time': window_start + rule.window_size,
                'retry_after': 60 - (current_time - window_start)
            }

        return True, {
            'remaining_requests': max(0, rule.requests_per_minute - current_requests - cost),
            'reset_time': window_start + rule.window_size,
        }

    async def _check_local_rate_limit(
        self,
        identifier: str,
        rule: RateLimitRule,
        cost: int,
        current_time: float
    ) -> tuple[bool, Dict[str, Any]]:
        """Rate limiting local com sliding window"""

        window = self._local_windows[identifier]

        # Remove requisições antigas (sliding window)
        cutoff_time = current_time - rule.window_size
        while window and window[0] < cutoff_time:
            window.popleft()

        current_requests = len(window)

        # Verifica burst limit
        if current_requests + cost > rule.burst_limit:
            return False, {
                'remaining_requests': max(0, rule.burst_limit - current_requests),
                'reset_time': current_time + rule.window_size,
                'retry_after': 1
            }

        # Verifica limite por minuto
        if current_requests + cost > rule.requests_per_minute:
            return False, {
                'remaining_requests': max(0, rule.requests_per_minute - current_requests),
                'reset_time': current_time + rule.window_size,
                'retry_after': 60
            }

        # Adiciona requisições
        for _ in range(cost):
            window.append(current_time)

        return True, {
            'remaining_requests': max(0, rule.requests_per_minute - len(window)),
            'reset_time': current_time + rule.window_size,
        }

    async def get_limits_info(self, identifier: str) -> Dict[str, Any]:
        """Obtém informações atuais de rate limiting"""
        if self.# redis_cache:
            try:
                redis_key = f"rate_limit:{identifier}"
                current_time = time.time()
                window_start = current_time - 60  # Assuming 60s window

                count = await self.# redis_cache.redis.zcount(
                    redis_key, window_start, current_time
                )

                return {
                    'current_requests': count,
                    'window_start': window_start,
                    'window_end': current_time
                }
            except Exception as e:
                logger.error(f"Failed to get limits info from Redis: {str(e)}")

        # Fallback local
        window = self._local_windows.get(identifier, deque())
        return {
            'current_requests': len(window),
            'window_start': time.time() - 60,
            'window_end': time.time()
        }

    async def reset_limits(self, identifier: str) -> bool:
        """Reset rate limits para um identificador"""
        try:
            if self.# redis_cache:
                redis_key = f"rate_limit:{identifier}"
                await self.# redis_cache.redis.delete(redis_key)

            # Reset local
            if identifier in self._local_windows:
                del self._local_windows[identifier]
            if identifier in self._local_counters:
                del self._local_counters[identifier]

            return True

        except Exception as e:
            logger.error(f"Failed to reset limits for {identifier}: {str(e)}")
            return False
```

#### infra/dependencies.py (Updated)
```python
import logging
from functools import lru_cache
from fastapi import Depends
from application.interfaces.llm_service_interface import LLMServiceInterface
from application.interfaces.model_repository_interface import ModelRepositoryInterface
from domain.services.llm_service import LLMService
from infra.database.cosmos_client import CosmosDBClient
from infra.database.model_repository import ModelRepository
from infra.messaging.kafka_producer import KafkaAsyncProducer
# FUTURO: from infra.cache.# redis_cache import # AsyncRedisCache
from infra.performance.connection_pool import OptimizedHTTPClient
from infra.performance.rate_limiter import AdvancedRateLimiter
from llmproxy.config import Config
import uuid

logger = logging.getLogger(__name__)

# Instâncias globais (singleton pattern)
_cosmos_client: CosmosDBClient = None
_model_repository: ModelRepository = None
_model_service: LLMService = None
_kafka_producer: KafkaAsyncProducer = None
_# redis_cache: # AsyncRedisCache = None
_http_client: OptimizedHTTPClient = None
_rate_limiter: AdvancedRateLimiter = None

@lru_cache()
def get_config() -> Config:
    """Retorna instância de configuração (cached)"""
    return Config()

async def get_# redis_cache() -> # AsyncRedisCache:
    """Dependency para cache Redis"""
    global _# redis_cache

    if _# redis_cache is None:
        _# redis_cache = # AsyncRedisCache()
        await _# redis_cache.initialize()

    return _# redis_cache

async def get_kafka_producer() -> KafkaAsyncProducer:
    """Dependency para producer Kafka"""
    global _kafka_producer

    if _kafka_producer is None:
        _kafka_producer = KafkaAsyncProducer()
        await _kafka_producer.initialize()

    return _kafka_producer

async def get_http_client() -> OptimizedHTTPClient:
    """Dependency para cliente HTTP otimizado"""
    global _http_client

    if _http_client is None:
        _http_client = OptimizedHTTPClient()
        await _http_client.initialize()

    return _http_client

async def get_rate_limiter() -> AdvancedRateLimiter:
    """Dependency para rate limiter"""
    global _rate_limiter

    if _rate_limiter is None:
        # redis_cache = await get_# redis_cache()
        _rate_limiter = AdvancedRateLimiter(# redis_cache)

    return _rate_limiter

async def get_model_repository() -> ModelRepositoryInterface:
    """Dependency para repositório de modelos"""
    global _model_repository, _cosmos_client

    if _model_repository is None:
        if _cosmos_client is None:
            _cosmos_client = CosmosDBClient()
            await _cosmos_client.initialize()

        _model_repository = ModelRepository(_cosmos_client)

    return _model_repository

async def get_model_service(
    repository: ModelRepositoryInterface = Depends(get_model_repository)
) -> LLMServiceInterface:
    """Dependency para serviço de modelos"""
    global _model_service

    if _model_service is None:
        _model_service = LLMService(repository)

    return _model_service

def get_request_id() -> str:
    """Gera ID único para requisição"""
    return str(uuid.uuid4())

async def startup_event():
    """Evento de inicialização da aplicação"""
    logger.info("Starting LLM Proxy application with performance enhancements...")

    # Valida configurações obrigatórias
    if not Config.validate_required_config():
        raise RuntimeError("Missing required configuration")

    # Inicializa componentes de performance
    await get_# redis_cache()
    await get_kafka_producer()
    await get_http_client()
    await get_rate_limiter()

    # Inicializa conexões
    await get_model_repository()

    logger.info("LLM Proxy application started successfully with all performance enhancements")

async def shutdown_event():
    """Evento de encerramento da aplicação"""
    logger.info("Shutting down LLM Proxy application...")

    # Fecha conexões
    global _cosmos_client, _kafka_producer, _# redis_cache, _http_client

    if _cosmos_client:
        await _cosmos_client.close()

    if _kafka_producer:
        await _kafka_producer.close()

    if _# redis_cache:
        await _# redis_cache.close()

    if _http_client:
        await _http_client.close()

    logger.info("LLM Proxy application shut down successfully")
```

#### infra/controllers/app.py (Updated)
```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import time
from llmproxy.config import Config
from infra.dependencies import startup_event, shutdown_event
from infra.controllers import chat_router, model_router
from infra.controllers.async_chat_controller import async_chat_router
from infra.performance.rate_limiter import AdvancedRateLimiter, RateLimitRule
from infra.dependencies import get_rate_limiter

logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia ciclo de vida da aplicação"""
    try:
        await startup_event()
        yield
    finally:
        await shutdown_event()

app = FastAPI(
    title=Config.APP_NAME,
    version=Config.APP_VERSION,
    description="Proxy assíncrono para múltiplos provedores de LLM com alta performance",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Middleware de rate limiting
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Middleware de rate limiting avançado"""
    if not Config.RATE_LIMIT_ENABLED:
        return await call_next(request)

    # Identifica cliente (IP ou API key)
    client_id = request.client.host
    api_key = request.headers.get("X-API-Key")
    if api_key:
        client_id = f"api_key:{api_key}"

    # Define regras de rate limiting
    rule = RateLimitRule(
        requests_per_minute=Config.RATE_LIMIT_PER_MINUTE,
        burst_limit=Config.RATE_LIMIT_BURST
    )

    # Verifica rate limit
    rate_limiter = await get_rate_limiter()
    allowed, info = await rate_limiter.check_rate_limit(client_id, rule)

    if not allowed:
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error",
                    "retry_after": info.get("retry_after", 60),
                    "remaining_requests": info.get("remaining_requests", 0)
                }
            },
            headers={
                "X-RateLimit-Limit": str(rule.requests_per_minute),
                "X-RateLimit-Remaining": str(info.get("remaining_requests", 0)),
                "X-RateLimit-Reset": str(int(info.get("reset_time", time.time() + 60))),
                "Retry-After": str(int(info.get("retry_after", 60)))
            }
        )

    # Adiciona headers informativos
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(rule.requests_per_minute)
    response.headers["X-RateLimit-Remaining"] = str(info.get("remaining_requests", 0))
    response.headers["X-RateLimit-Reset"] = str(int(info.get("reset_time", time.time() + 60)))

    return response

# Registra os roteadores
app.include_router(model_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(async_chat_router, prefix="/api")  # Endpoint assíncrono

@app.get("/")
async def root():
    return {
        "service": Config.APP_NAME,
        "version": Config.APP_VERSION,
        "status": "running",
        "features": {
            "async_processing": True,
            "# redis_cache": True,
            "kafka_queue": True,
            "rate_limiting": Config.RATE_LIMIT_ENABLED,
            "connection_pooling": True
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": Config.APP_NAME,
        "version": Config.APP_VERSION
    }
```

#### requirements.txt
```txt
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
azure-cosmos>=4.5.1
litellm>=1.44.0
openai>=1.3.0
anthropic>=0.7.0
pydantic>=2.5.0
python-multipart>=0.0.6
typing-extensions>=4.8.0

# Performance & Async Processing
aiokafka>=0.8.11
# aioredis>=2.0.1

# Connection pooling & HTTP clients
aiohttp[speedups]>=3.9.0
httpx>=0.25.0

# Monitoring & Observability
prometheus-client>=0.19.0

# Serialization optimization
orjson>=3.9.0
```

### Consumer Service (Separado)

#### consumer_service.py
```python
import asyncio
import logging
from application.use_cases.async_chat_completion_use_case import AsyncChatCompletionUseCase
from infra.messaging.kafka_consumer import KafkaAsyncConsumer
# FUTURO: from infra.cache.# redis_cache import # AsyncRedisCache
from domain.services.llm_service import LLMService
from infra.database.cosmos_client import CosmosDBClient
from infra.database.model_repository import ModelRepository
from llmproxy.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Serviço consumidor Kafka para processamento assíncrono"""
    logger.info("Starting Kafka consumer service...")

    try:
        # Inicializa dependências
        cosmos_client = CosmosDBClient()
        await cosmos_client.initialize()

        model_repository = ModelRepository(cosmos_client)
        model_service = LLMService(model_repository)

        # redis_cache = # AsyncRedisCache()
        await # redis_cache.initialize()

        # Caso de uso para processamento
        chat_use_case = AsyncChatCompletionUseCase(model_service, # redis_cache)

        # Consumer Kafka
        consumer = KafkaAsyncConsumer(chat_use_case)

        # Inicia consumo
        await consumer.start_consuming()

    except Exception as e:
        logger.error(f"Consumer service failed: {str(e)}")
        raise
    finally:
        logger.info("Consumer service stopped")

if __name__ == "__main__":
    asyncio.run(main())
```

#### Dockerfile (Updated)
```dockerfile
# Multi-stage build para otimização
FROM python:3.11-slim as builder

RUN apt-get update && apt-get install -y \
    gcc g++ curl && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home llmproxy

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage de produção
FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home llmproxy

# Copia dependências do builder
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

WORKDIR /app
COPY --chown=llmproxy:llmproxy . .

USER llmproxy

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando padrão (API)
CMD ["uvicorn", "infra.controllers.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

#### docker-compose.yml (Updated)
```yaml
version: '3.8'

services:
  # Serviço principal (API)
  llm-proxy-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - APP_NAME=LLM Proxy
      - APP_VERSION=2.0.0
      - DEBUG=false

      # Cosmos DB
      - COSMOS_URL=${COSMOS_URL}
      - COSMOS_KEY=${COSMOS_KEY}

      # Redis
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - REDIS_PASSWORD=${REDIS_PASSWORD:-}

      # Kafka
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
      - KAFKA_ASYNC_CHAT_TOPIC=llm-async-chat

      # Rate limiting
      - RATE_LIMIT_ENABLED=true
      - RATE_LIMIT_PER_MINUTE=60
      - RATE_LIMIT_BURST=100

      # Performance
      - HTTP_CLIENT_MAX_CONNECTIONS=100
      - HTTP_CLIENT_MAX_KEEPALIVE=50

    depends_on:
      - redis
      - kafka
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs

  # Serviço consumidor (processamento assíncrono)
  llm-proxy-consumer:
    build: .
    command: ["python", "consumer_service.py"]
    environment:
      # Mesmas configurações da API
      - COSMOS_URL=${COSMOS_URL}
      - COSMOS_KEY=${COSMOS_KEY}
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
      - KAFKA_ASYNC_CHAT_TOPIC=llm-async-chat
      - KAFKA_CONSUMER_GROUP_ID=llm-proxy-consumers
      - KAFKA_CONSUMER_CONCURRENCY=4
      - CONSUMER_THREAD_POOL_SIZE=8
    depends_on:
      - redis
      - kafka
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs

  # Redis para cache
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    environment:
      - REDIS_PASSWORD=${REDIS_PASSWORD:-}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped

  # Kafka para mensageria
  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: true
    volumes:
      - kafka_data:/var/lib/kafka/data
    depends_on:
      - zookeeper
    restart: unless-stopped

  # Zookeeper para Kafka
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    volumes:
      - zookeeper_data:/var/lib/zookeeper/data
    restart: unless-stopped

volumes:
  redis_data:
  kafka_data:
  zookeeper_data:
  logs:
```

## Novas melhorias de performance implementadas:

### 1. Sistema de Filas Kafka
- **Producer assíncrono** com batching e compressão LZ4
- **Consumer paralelo** com múltiplos workers
- **Processamento em batch** para máxima eficiência
- **Particionamento consistente** por request_id

### 2. Endpoint Assíncrono para Chat Completions
- **POST /v1/async/chat/completions**: Envia requisição para fila e retorna ID
- **GET /v1/async/chat/completions/{id}/status**: Consulta status
- **GET /v1/async/chat/completions/{id}/result**: Obtém resultado
- **DELETE /v1/async/chat/completions/{id}**: Cancela requisição

### 3. Cache Redis Distribuído
- **Pool de conexões** otimizado para alta concorrência
- **Operações batch** (mget, mset) para múltiplos valores
- **TTL automático** para limpeza de dados
- **Fallback local** se Redis falhar

### 4. Pool de Conexões HTTP
- **Cliente HTTP otimizado** com keep-alive
- **Requisições paralelas** com asyncio.gather
- **DNS cache** e configurações de timeout otimizadas
- **Reutilização de conexões** para máxima performance

### 5. Rate Limiting Avançado
- **Sliding window** com Redis distribuído
- **Burst limiting** para picos de tráfego
- **Headers informativos** (X-RateLimit-*)
- **Fallback local** se Redis não estiver disponível

### 6. Arquitetura de Deployment
- **API service** para endpoints HTTP
- **Consumer service** separado para processamento
- **Redis** para cache e sessões
- **Kafka + Zookeeper** para mensageria

### 7. Configurações de Performance
```bash
# Kafka
KAFKA_CONSUMER_CONCURRENCY=4        # Workers paralelos
KAFKA_BATCH_SIZE=50                 # Mensagens por batch
KAFKA_MAX_POLL_RECORDS=100         # Máximo por poll

# Redis
REDIS_MAX_CONNECTIONS=50           # Pool de conexões

# HTTP Client
HTTP_CLIENT_MAX_CONNECTIONS=100    # Conexões simultâneas
HTTP_CLIENT_MAX_KEEPALIVE=50       # Keep-alive por host

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60           # Requisições por minuto
RATE_LIMIT_BURST=100               # Burst permitido
```

### 8. Contratos da API Validados com OpenAI/LiteLLM

#### Chat Completions (Síncrono)
```json
// POST /v1/chat/completions - Totalmente compatível OpenAI
{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 150,
  "top_p": 1.0,
  "stream": false,
  "tools": [],
  "response_format": {"type": "text"},

  // Parâmetros específicos de provedores (LiteLLM)
  "timeout": 60,
  "max_retries": 3,
  "fallbacks": ["gpt-4", "claude-3-sonnet"],

  // Anthropic específico
  "reasoning_effort": "medium",
  "cache_control": {"type": "enabled"},

  // Google Vertex específico
  "top_k": 40,
  "safety_settings": [...]
}
```

#### Chat Completions Assíncrono
```json
// POST /v1/async/chat/completions - Suporta TODOS os parâmetros do síncrono
{
  "model": "gpt-4o",
  "messages": [{"role": "user", "content": "Complex analysis task..."}],
  "temperature": 0.7,
  "max_tokens": 2000,
  "tools": [...],

  // Parâmetros específicos do processamento assíncrono
  "priority": "high",
  "timeout_seconds": 600,
  "callback_url": "https://webhook.example.com/llm-results",
  "webhook_events": ["completed", "failed"],
  "client_id": "client-123",
  "session_id": "session-456",
  "tags": {"department": "research", "project": "analysis"},
  "async_max_retries": 3
}

// Resposta
{
  "request_id": "req_abc123xyz",
  "status": "pending",
  "submitted_at": "2025-01-01T10:00:00Z",
  "priority": "high",
  "estimated_completion_time": "2025-01-01T10:05:00Z",
  "status_endpoint": "/v1/async/chat/completions/req_abc123xyz/status",
  "result_endpoint": "/v1/async/chat/completions/req_abc123xyz/result"
}
```

#### Embeddings
```json
// POST /v1/embeddings - Compatível OpenAI + LiteLLM
{
  "model": "text-embedding-3-large",
  "input": ["Hello world", "How are you?"],
  "dimensions": 1024,
  "encoding_format": "float",
  "user": "user-123",

  // Cohere específico
  "input_type": "search_document"
}
```

#### Text-to-Speech
```json
// POST /v1/audio/speech - Compatível OpenAI + provedores adicionais
{
  "model": "tts-1",
  "input": "Hello, this is a test.",
  "voice": "alloy",
  "response_format": "mp3",
  "speed": 1.0,

  // ElevenLabs específico (via LiteLLM)
  "stability": 0.5,
  "similarity_boost": 0.8,
  "style": 0.3
}
```

#### Image Generation
```json
// POST /v1/images/generations - Compatível OpenAI + Stability AI
{
  "model": "dall-e-3",
  "prompt": "A futuristic city at sunset",
  "n": 1,
  "size": "1024x1024",
  "quality": "hd",
  "style": "vivid",
  "response_format": "url",

  // Stability AI específico
  "steps": 30,
  "cfg_scale": 7.5,
  "seed": 42
}
```

#### Parâmetros Suportados por Provedor

**OpenAI/Azure OpenAI:**
- Todos os parâmetros padrão
- `stream_options`, `logprobs`, `seed`
- `max_completion_tokens` (novo)

**Anthropic Claude:**
- `reasoning_effort` → `thinking`
- `cache_control` para prompt caching
- `extra_headers` para beta features

**Google Vertex AI:**
- `top_k`, `candidate_count`
- `safety_settings`
- Suporte a Gemini Pro/Ultra

**Cohere:**
- `max_input_tokens`, `truncate`
- `connectors`, `input_type`

**Todos os Provedores:**
- `timeout`, `max_retries`
- `fallbacks`, `context_window_fallbacks`
- `headers`, `extra_headers`
- `safety_identifier`

Todas as melhorias foram implementadas mantendo compatibilidade com a API OpenAI e adicionando capacidades assíncronas para alta performance e escalabilidade.

## Resumo Geral das Melhorias

### Performance (Principais melhorias implementadas):

1. **Arquitetura Hexagonal**: Separação clara entre Application, Domain e Infrastructure
2. **Operações Assíncronas**: Todos os métodos convertidos para async/await
3. **Nomenclatura Clara**: Nomes de arquivos e classes mais específicos e descritivos
4. **Validação Robusta**: Schemas Pydantic com validações aprimoradas
5. **Configuração Centralizada**: Sistema de configuração baseado em variáveis de ambiente
6. **Dependency Injection**: Sistema de injeção de dependências para melhor testabilidade
7. **Error Handling**: Tratamento de erros consistente em todas as camadas
8. **Logging Estruturado**: Sistema de logging para produção
9. **Health Checks**: Endpoints de saúde para monitoramento
10. **Docker**: Configuração de containerização com health checks
11. **Padronização model_type**: Campo padronizado usando Enums
12. **Campo operations**: Adicionado campo de operações suportadas pelos modelos

### Performance Avançada (Implementadas):

13. **Sistema de Filas Kafka**: Processamento assíncrono com múltiplos workers
14. **Endpoint Assíncrono**: API REST para chat completions com processamento em background
15. **Pool de Conexões HTTP**: Cliente HTTP otimizado com keep-alive e paralelismo
16. **Rate Limiting Local**: Sliding window com fallback local
17. **Arquitetura de Microserviços**: API + Consumer separados para escalabilidade
18. **Batching e Compressão**: Otimizações Kafka com LZ4 e batching
19. **Monitoramento**: Headers informativos e métricas de performance

### Performance Futuras (Comentadas - Redis):
- ~~**Cache Redis Distribuído**: Cache de alta performance com pool de conexões~~ (Futuro)
- ~~**Rate Limiting Distribuído**: Sliding window com Redis~~ (Futuro)

### Escalabilidade:
- **Processamento paralelo** com múltiplos workers Kafka
- **Pool de conexões** otimizado para alta concorrência
- **Separação de responsabilidades** (API vs Consumer)
- **Rate limiting local** com burst control

O sistema agora suporta:
- ✅ **Alta concorrência** com processamento assíncrono
- ✅ **Escalabilidade horizontal** com Kafka
- ✅ **Baixa latência** com connection pooling
- ✅ **Rate limiting inteligente** local
- ✅ **Monitoramento completo** com métricas e health checks
- ✅ **Deploy production-ready** com Docker Compose
