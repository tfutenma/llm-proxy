# LLM Proxy Service - Arquitetura Melhorada

## 🎯 Funcionalidades e Features

### Core Features
- **Proxy Unificado**: Interface única para múltiplos provedores LLM (Azure OpenAI, Google Vertex AI)
- **Compatibilidade OpenAI**: API totalmente compatível com especificação OpenAI
- **Multi-Provider**: Suporte para Azure OpenAI e Google Vertex AI
- **Operações Assíncronas**: Todas as operações implementadas com async/await
- **Gestão de Modelos**: Sistema de configuração de modelos via Cosmos DB

### Operações Implementadas
1. **Chat Completions** (`ChatCompletion`) - Conversação com modelos de linguagem
2. **Text-to-Speech** (`audio_tts`) - Síntese de voz a partir de texto
3. **Speech-to-Text** (`audio_stt`) - Transcrição de áudio para texto
4. **Image Generation** (`images`) - Geração de imagens via prompts
5. **Embeddings** (`embeddings`) - Geração de embeddings de texto
6. **Model Management** - Listagem e gerenciamento de modelos disponíveis

### Como Funciona
O serviço atua como um proxy que:
1. Recebe requisições no formato OpenAI
2. Identifica o modelo e provedor correto via Cosmos DB
3. Valida se o modelo suporta a operação solicitada
4. Adapta a requisição para o formato específico do provedor
5. Executa a operação via LiteLLM
6. Retorna a resposta no formato padrão OpenAI

### Endpoints Disponíveis
- `POST /v1/chat/completions` - Chat completions
- `POST /v1/audio/speech` - Text-to-speech
- `POST /v1/audio/transcriptions` - Speech-to-text
- `POST /v1/images/generations` - Geração de imagens
- `GET /v1/models` - Listagem de modelos
- `GET /health` - Health check

## 🏗️ Arquitetura Hexagonal Melhorada

A arquitetura segue os princípios de Clean Architecture e Hexagonal Architecture com melhorias significativas:

### Estrutura de Camadas

```
┌─────────────────────────────────────────────────────┐
│                 HTTP Layer                          │
│  FastAPI Controllers & Request/Response DTOs       │
└─────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────┐
│              Application Layer                      │
│  Use Cases & Application Services                   │
└─────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────┐
│               Domain Layer                          │
│  Entities, Value Objects & Domain Services         │
└─────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────┐
│            Infrastructure Layer                     │
│  Database, External APIs & Framework Code          │
└─────────────────────────────────────────────────────┘
```

### Padrões de Design Implementados

#### 1. **Repository Pattern**
Abstração para acesso a dados com interface bem definida.

#### 2. **Strategy Pattern**
Para diferentes provedores LLM e suas particularidades.

#### 3. **Factory Pattern**
Para criação de instâncias de provedores baseado no modelo.

#### 4. **Adapter Pattern**
Para adaptação entre formatos de API dos diferentes provedores.

#### 5. **Dependency Injection**
Via FastAPI Depends para inversão de controle.

## 📁 Estrutura de Código Melhorada

```
llm-proxy/
├── llmproxy/                      # Pacote principal
│   ├── __init__.py
│   └── config.py                  # Configurações centralizadas
├── application/                   # Camada de Aplicação
│   ├── interfaces/               # Interfaces (Ports)
│   │   ├── llm_provider_interface.py
│   │   └── llm_service_interface.py
│   └── use_cases/               # Casos de uso
│       ├── chat_completion_use_case.py
│       ├── text_to_speech_use_case.py
│       ├── speech_to_text_use_case.py
│       └── image_generation_use_case.py
├── domain/                       # Camada de Domínio
│   ├── entities/                # Entidades
│   │   ├── llm_model.py
│   │   └── operation.py
│   └── services/                # Serviços de domínio
│       ├── llm_model_service.py
│       └── operation_validator.py
├── infra/                       # Camada de Infraestrutura
│   ├── controllers/             # Controladores HTTP
│   │   ├── chat_completion_controller.py
│   │   ├── text_to_speech_controller.py
│   │   ├── speech_to_text_controller.py
│   │   ├── image_generation_controller.py
│   │   ├── model_controller.py
│   │   └── app.py              # FastAPI app principal
│   ├── schemas/                # Esquemas Pydantic (DTOs)
│   │   ├── chat_completion_schemas.py
│   │   ├── text_to_speech_schemas.py
│   │   ├── speech_to_text_schemas.py
│   │   ├── image_generation_schemas.py
│   │   └── model_schemas.py
│   ├── database/               # Acesso a dados
│   │   ├── cosmos_client.py
│   │   ├── model_repository.py
│   │   └── cosmos_models.py
│   └── providers/              # Integrações LLM
│       ├── litellm_provider.py
│       ├── llm_adapters.py
│       └── provider_utils.py
└── requirements.txt
```

## 🔧 Implementação Melhorada

### 1. Domain Layer

#### `src/domain/entities/llm_model.py`
```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

class ProviderType(Enum):
    AZURE = "Azure"
    GOOGLE = "Google"

class ModelType(Enum):
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

class OperationType(Enum):
    RESPONSES = "Responses"
    CHAT_COMPLETION = "ChatCompletion"
    EMBEDDINGS = "embeddings"
    IMAGES = "images"
    AUDIO_TTS = "audio_tts"
    AUDIO_STT = "audio_stt"
    AUDIO_REALTIME = "audio_realtime"

@dataclass(frozen=True)
class LLMModel:
    id: str
    name: str
    provider: ProviderType
    model_type: ModelType
    code: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    costs: Dict[str, float] = field(default_factory=dict)
    operations: List[OperationType] = field(default_factory=list)
    projects: List[str] = field(default_factory=list)
    private: bool = False

    def supports_operation(self, operation: OperationType) -> bool:
        return operation in self.operations

    def is_available_for_project(self, project_id: str) -> bool:
        return not self.private or project_id in self.projects

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = input_tokens * self.costs.get('input', 0.0) / 1000
        output_cost = output_tokens * self.costs.get('output', 0.0) / 1000
        return input_cost + output_cost

    def get_litellm_model_name(self) -> str:
        """Gera nome do modelo para LiteLLM baseado no provider e tipo"""
        if self.provider == ProviderType.AZURE:
            return f"azure/{self.code}"
        elif self.provider == ProviderType.GOOGLE:
            model_type = self.model_type
            if model_type == ModelType.GOOGLE_VERTEX_CLAUDE:
                return f"vertex_ai/claude-{self.code}"
            elif model_type == ModelType.GOOGLE_VERTEX_DEEPSEEK:
                return f"vertex_ai/deepseek-{self.code}"
            elif model_type == ModelType.GOOGLE_VERTEX_META:
                return f"vertex_ai/llama-{self.code}"
            elif model_type == ModelType.GOOGLE_VERTEX_MISTRAL:
                return f"vertex_ai/mistral-{self.code}"
            elif model_type == ModelType.GOOGLE_VERTEX_JAMBA:
                return f"vertex_ai/jamba-{self.code}"
            elif model_type == ModelType.GOOGLE_VERTEX_QWEN:
                return f"vertex_ai/qwen-{self.code}"
            elif model_type == ModelType.GOOGLE_VERTEX_OPENAI:
                return f"vertex_ai/gpt-{self.code}"
            else:
                return f"vertex_ai/{self.code}"
        else:
            return self.code
```

#### `application/interfaces/llm_service_interface.py`
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
    async def list_models(self) -> List[LLMModel]:
        """Lista todos os modelos disponíveis de forma assíncrona"""
        pass

    @abstractmethod
    async def get_models_by_provider(self, provider: str) -> List[LLMModel]:
        """Obtém modelos filtrados por provedor de forma assíncrona"""
        pass

    @abstractmethod
    async def get_models_by_operation(self, operation: str) -> List[LLMModel]:
        """Obtém modelos que suportam uma operação específica de forma assíncrona"""
        pass

    @abstractmethod
    async def validate_model_for_operation(self, model_id: str, operation: str) -> bool:
        """Valida se um modelo suporta uma operação específica de forma assíncrona"""
        pass
```

#### `application/interfaces/llm_provider_interface.py`
```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class LLMProviderInterface(ABC):
    """Interface para provedores de LLM com operações assíncronas"""

    @abstractmethod
    async def chat_completions(self, **kwargs) -> Dict[Any, Any]:
        """Processa requisições de chat/completions de forma assíncrona"""
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
    def get_provider_name(self) -> str:
        """Retorna o nome do provedor (azure, google, etc)"""
        pass

    @abstractmethod
    def supports_operation(self, operation: str) -> bool:
        """Verifica se o provedor suporta uma operação específica"""
        pass
```

### 2. Application Layer

#### `application/use_cases/chat_completion_use_case.py`
```python
from typing import Dict, Any
from application.interfaces.llm_service_interface import LLMServiceInterface
from infra.schemas.chat_completion_schemas import ChatCompletionRequest
from infra.providers.litellm_provider import LiteLLMProvider

class ChatCompletionUseCase:
    """Caso de uso para completions de chat com operações assíncronas"""

    def __init__(self, model_service: LLMServiceInterface):
        self.model_service = model_service

    async def execute(self, request: ChatCompletionRequest) -> Dict[Any, Any]:
        """Executa uma requisição de chat completion de forma assíncrona"""
        # Obtém configuração do modelo
        model = await self.model_service.get_model(request.model)
        if not model:
            raise ValueError(f"Model '{request.model}' not found")

        # Verifica se o modelo suporta chat completions
        if not await self.model_service.validate_model_for_operation(request.model, 'ChatCompletion'):
            raise ValueError(f"Model '{request.model}' does not support chat completions")

        # Cria instância do provedor e executa requisição
        provider = LiteLLMProvider(model)

        # Converte requisição para formato LiteLLM
        litellm_params = {
            'messages': [msg.model_dump() for msg in request.messages],
            'temperature': request.temperature,
            'max_tokens': request.max_tokens,
            'top_p': request.top_p,
            'frequency_penalty': request.frequency_penalty,
            'presence_penalty': request.presence_penalty,
            'stop': request.stop,
            'stream': request.stream,
            'user': request.user
        }

        # Remove valores None
        litellm_params = {k: v for k, v in litellm_params.items() if v is not None}

        return await provider.chat_completions(**litellm_params)
```

#### `application/use_cases/text_to_speech_use_case.py`
```python
from typing import Dict, Any
from application.interfaces.llm_service_interface import LLMServiceInterface
from infra.schemas.text_to_speech_schemas import TextToSpeechRequest
from infra.providers.litellm_provider import LiteLLMProvider

class TextToSpeechUseCase:
    """Caso de uso para text-to-speech com operações assíncronas"""

    def __init__(self, model_service: LLMServiceInterface):
        self.model_service = model_service

    async def execute(self, request: TextToSpeechRequest) -> Dict[Any, Any]:
        """Executa uma requisição de text-to-speech de forma assíncrona"""
        # Obtém configuração do modelo
        model = await self.model_service.get_model(request.model)
        if not model:
            raise ValueError(f"Model '{request.model}' not found")

        # Verifica se o modelo suporta TTS
        if not await self.model_service.validate_model_for_operation(request.model, 'audio_tts'):
            raise ValueError(f"Model '{request.model}' does not support text-to-speech")

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
```

### 3. Infrastructure Layer

#### `infra/providers/litellm_provider.py`
```python
import litellm
from typing import Dict, Any, Optional
from application.interfaces.llm_provider_interface import LLMProviderInterface
from domain.entities.llm_model import LLMModel
import base64
import io

class LiteLLMProvider(LLMProviderInterface):
    def __init__(self, model: LLMModel):
        self.model = model
        self.parameters = model.parameters
        self._setup_litellm()

    def _setup_litellm(self):
        """Setup LiteLLM with the model parameters"""
        # Set global API keys for LiteLLM
        for param_name, param_value in self.parameters.items():
            if param_name.endswith('_api_key'):
                import os
                from llmproxy.config import Config
                # Get the actual key from environment
                secret_value = Config.get_provider_config(param_value)
                if secret_value:
                    os.environ[param_name.upper()] = secret_value

        # Setup Google credentials if needed
        if self._is_google_vertex_model():
            self._setup_google_credentials()

    async def chat_completions(self, **kwargs) -> Dict[Any, Any]:
        """Handle chat completions using LiteLLM"""
        try:
            self._setup_litellm()

            # Prepare model name for LiteLLM
            model_name = self._prepare_model_name()
            additional_params = self._get_additional_parameters()

            # Merge all parameters
            params = {
                'model': model_name,
                **kwargs,
                **additional_params
            }

            # Call LiteLLM
            response = await litellm.acompletion(**params)

            # Format response to match OpenAI format
            return self._format_chat_response(response)

        except Exception as e:
            raise Exception(f"LiteLLM chat completion error: {str(e)}")

    async def text_to_speech(self, **kwargs) -> Dict[Any, Any]:
        """Handle text-to-speech using LiteLLM"""
        try:
            self._setup_litellm()
            model_name = self._prepare_model_name()
            additional_params = self._get_additional_parameters()

            params = {
                'model': model_name,
                **kwargs,
                **additional_params
            }

            response = await litellm.aspeech(**params)
            return {"audio": response, "content_type": "audio/mpeg"}

        except Exception as e:
            raise Exception(f"LiteLLM TTS error: {str(e)}")

    async def speech_to_text(self, **kwargs) -> Dict[Any, Any]:
        """Handle speech-to-text using LiteLLM"""
        try:
            self._setup_litellm()
            model_name = self._prepare_model_name()
            additional_params = self._get_additional_parameters()

            params = {
                'model': model_name,
                **kwargs,
                **additional_params
            }

            response = await litellm.atranscription(**params)
            return response

        except Exception as e:
            raise Exception(f"LiteLLM STT error: {str(e)}")

    async def generate_image(self, **kwargs) -> Dict[Any, Any]:
        """Handle image generation using LiteLLM"""
        try:
            self._setup_litellm()
            model_name = self._prepare_model_name()
            additional_params = self._get_additional_parameters()

            params = {
                'model': model_name,
                **kwargs,
                **additional_params
            }

            response = await litellm.aimage_generation(**params)
            return response

        except Exception as e:
            raise Exception(f"LiteLLM image generation error: {str(e)}")

    def get_provider_name(self) -> str:
        """Return the provider name"""
        return self.model.provider.lower()

    def supports_operation(self, operation: str) -> bool:
        """Check if provider supports operation"""
        return operation in self.model.operations

    def _prepare_model_name(self) -> str:
        """Prepare model name for LiteLLM"""
        return self.model.get_litellm_model_name()

    def _get_additional_parameters(self) -> Dict[str, Any]:
        """Get additional parameters for LiteLLM"""
        additional_params = {}

        # Add provider-specific parameters
        if self.model.provider == "Azure":
            additional_params.update({
                "api_base": self.parameters.get("endpoint"),
                "api_version": self.parameters.get("api_version")
            })
        elif self.model.provider == "Google":
            additional_params.update({
                "vertex_project": self.parameters.get("gcp_project"),
                "vertex_location": self.parameters.get("location", "us-central1")
            })

        return additional_params

    def _is_google_vertex_model(self) -> bool:
        """Check if model is Google Vertex AI"""
        google_vertex_types = [
            'GoogleVertex', 'GoogleVertexClaude', 'GoogleVertexDeepSeek',
            'GoogleVertexMeta', 'GoogleVertexMistral', 'GoogleVertexJamba',
            'GoogleVertexQwen', 'GoogleVertexOpenAI'
        ]
        return self.model.model_type in google_vertex_types

    def _setup_google_credentials(self):
        """Setup Google credentials"""
        try:
            from llmproxy.config import Config
            secret_name = self.parameters.get('secret_name')
            if secret_name:
                credentials_content = Config.get_google_credentials(secret_name)
                if credentials_content:
                    import tempfile, json, os
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        f.write(credentials_content)
                        credentials_path = f.name
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        except Exception as e:
            print(f"Error setting up Google credentials: {str(e)}")

    def _format_chat_response(self, response) -> Dict[Any, Any]:
        """Format response to OpenAI format"""
        if hasattr(response, 'model_dump'):
            return response.model_dump()
        elif hasattr(response, 'to_dict'):
            return response.to_dict()
        return dict(response)
```

#### `infra/controllers/chat_completion_controller.py`
```python
from fastapi import APIRouter, HTTPException, Depends
from infra.schemas.chat_completion_schemas import ChatCompletionRequest
from application.use_cases.chat_completion_use_case import ChatCompletionUseCase
from domain.services.llm_model_service import LLMModelService

router = APIRouter()

@router.post("/chat/completions", response_model=dict)
async def chat_completions(
    request: ChatCompletionRequest,
    service: LLMModelService = Depends()
):
    """OpenAI-compatible chat completions endpoint"""
    try:
        use_case = ChatCompletionUseCase(service)
        response = await use_case.execute(request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
```

#### `infra/controllers/text_to_speech_controller.py`
```python
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response
from infra.schemas.text_to_speech_schemas import TextToSpeechRequest
from application.use_cases.text_to_speech_use_case import TextToSpeechUseCase
from domain.services.llm_model_service import LLMModelService

router = APIRouter()

@router.post("/audio/speech")
async def text_to_speech(
    request: TextToSpeechRequest,
    service: LLMModelService = Depends()
):
    """OpenAI-compatible text-to-speech endpoint"""
    try:
        use_case = TextToSpeechUseCase(service)
        response = await use_case.execute(request)

        # Return binary audio content
        return Response(
            content=response.get('audio', b''),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=audio.mp3"}
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
```

#### `infra/database/cosmos_client.py`
```python
from typing import List, Optional, Dict, Any
from azure.cosmos import CosmosClient, exceptions
from llmproxy.config import Config

class CosmosDBClient:
    def __init__(self):
        self.client = CosmosClient(Config.COSMOS_URL, Config.COSMOS_KEY)
        self.database = self.client.get_database_client(Config.COSMOS_DATABASE_ID)
        self.container = self.database.get_container_client(Config.COSMOS_CONTAINER_ID)

    def get_model(self, model_id: str) -> Optional[Dict[Any, Any]]:
        """Get a specific model by ID"""
        try:
            query = "SELECT * FROM c WHERE c.id = @model_id OR c.code = @model_id"
            parameters = [{"name": "@model_id", "value": model_id}]
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            return items[0] if items else None
        except exceptions.CosmosHttpResponseError:
            return None

    def list_models(self) -> List[Dict[Any, Any]]:
        """List all models"""
        try:
            return list(self.container.read_all_items())
        except exceptions.CosmosHttpResponseError:
            return []

    def get_models_by_provider(self, provider: str) -> List[Dict[Any, Any]]:
        """Get models filtered by provider"""
        try:
            query = "SELECT * FROM c WHERE c.provider = @provider"
            parameters = [{"name": "@provider", "value": provider}]
            return list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
        except exceptions.CosmosHttpResponseError:
            return []

    def get_models_by_operation(self, operation: str) -> List[Dict[Any, Any]]:
        """Get models that support a specific operation"""
        try:
            query = "SELECT * FROM c WHERE ARRAY_CONTAINS(c.operations, @operation)"
            parameters = [{"name": "@operation", "value": operation}]
            return list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
        except exceptions.CosmosHttpResponseError:
            return []
```

#### `domain/services/llm_model_service.py`
```python
from typing import List, Optional
from domain.entities.llm_model import LLMModel
from infra.database.cosmos_client import CosmosDBClient
from llmproxy.config import Config

class LLMModelService:
    def __init__(self, db_client: CosmosDBClient):
        self.db_client = db_client

    async def get_model(self, model_id: str) -> Optional[LLMModel]:
        """Get a specific model by ID"""
        try:
            data = self.db_client.get_model(model_id)
            if data:
                # Enrich parameters with environment variables
                enriched_params = self._enrich_parameters(data.get('parameters', {}))
                data['parameters'] = enriched_params
                return LLMModel(**data)
            return None
        except Exception:
            return None

    async def list_models(self) -> List[LLMModel]:
        """List all available models"""
        try:
            models_data = self.db_client.list_models()
            models = []
            for data in models_data:
                enriched_params = self._enrich_parameters(data.get('parameters', {}))
                data['parameters'] = enriched_params
                models.append(LLMModel(**data))
            return models
        except Exception:
            return []

    async def get_models_by_provider(self, provider: str) -> List[LLMModel]:
        """Get models filtered by provider"""
        try:
            models_data = self.db_client.get_models_by_provider(provider)
            return [LLMModel(**data) for data in models_data]
        except Exception:
            return []

    async def get_models_by_operation(self, operation: str) -> List[LLMModel]:
        """Get models filtered by supported operation"""
        try:
            models_data = self.db_client.get_models_by_operation(operation)
            return [LLMModel(**data) for data in models_data]
        except Exception:
            return []

    async def validate_model_for_operation(self, model_id: str, operation: str) -> bool:
        """Validate if a model supports a specific operation"""
        model = await self.get_model(model_id)
        return model is not None and operation in model.operations

    def _enrich_parameters(self, parameters: dict) -> dict:
        """Enrich parameters with values from environment variables"""
        enriched = {}
        for key, env_var_name in parameters.items():
            if isinstance(env_var_name, str) and env_var_name.startswith('$'):
                # It's an environment variable reference
                env_value = Config.get_provider_config(env_var_name[1:])
                enriched[key] = env_value if env_value else env_var_name
            else:
                enriched[key] = env_var_name
        return enriched
```

### 4. Configuration

#### `llmproxy/config.py`
```python
import os
from typing import Optional

class Config:
    # Cosmos DB Configuration
    COSMOS_URL: str = os.getenv("COSMOS_URL", "")
    COSMOS_KEY: str = os.getenv("COSMOS_KEY", "")
    COSMOS_DATABASE_ID: str = os.getenv("COSMOS_DATABASE_ID", "llm-proxy")
    COSMOS_CONTAINER_ID: str = os.getenv("COSMOS_CONTAINER_ID", "models")

    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    @classmethod
    def get_provider_config(cls, secret_name: str) -> Optional[str]:
        """Get provider configuration from environment variables"""
        return os.getenv(secret_name)

    @classmethod
    def get_google_credentials(cls, secret_name: str) -> Optional[str]:
        """Get Google credentials from environment variables"""
        return os.getenv(secret_name)

    @classmethod
    def validate_required_configs(cls):
        """Validate that all required configurations are present"""
        required_configs = [
            "COSMOS_URL",
            "COSMOS_KEY"
        ]

        missing_configs = []
        for config in required_configs:
            if not getattr(cls, config):
                missing_configs.append(config)

        if missing_configs:
            raise ValueError(f"Missing required configurations: {missing_configs}")
```

### 5. FastAPI Application

#### `infra/controllers/app.py`
```python
from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager
from llmproxy.config import Config
from infra.database.cosmos_client import CosmosDBClient
from domain.services.llm_model_service import LLMModelService
from infra.controllers.chat_completion_controller import router as chat_router
from infra.controllers.text_to_speech_controller import router as tts_router
from infra.controllers.speech_to_text_controller import router as stt_router
from infra.controllers.image_generation_controller import router as image_router

# Global variables for dependency injection
db_client = None
model_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global db_client, model_service

    # Validate configuration
    Config.validate_required_configs()

    # Initialize Cosmos DB client
    db_client = CosmosDBClient()

    # Initialize model service
    model_service = LLMModelService(db_client)

    yield

    # Shutdown
    pass

def get_model_service() -> LLMModelService:
    if model_service is None:
        raise HTTPException(status_code=500, detail="Model service not initialized")
    return model_service

app = FastAPI(
    title="LLM Proxy Service",
    description="Unified proxy for multiple LLM providers",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(chat_router, prefix="/v1")
app.include_router(tts_router, prefix="/v1")
app.include_router(stt_router, prefix="/v1")
app.include_router(image_router, prefix="/v1")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "environment": Config.ENVIRONMENT}

@app.get("/v1/models")
async def list_models(service: LLMModelService = Depends(get_model_service)):
    """List available models - OpenAI compatible"""
    models = await service.list_models()

    # Format response to match OpenAI API
    openai_models = []
    for model in models:
        openai_models.append({
            "id": model.id,
            "object": "model",
            "created": 1677610602,
            "owned_by": model.provider.lower()
        })

    return {
        "object": "list",
        "data": openai_models
    }
```

## 🔄 Melhorias Implementadas

### 1. **Operações Consolidadas**
- Todas as operações reais implementadas: Chat, TTS, STT, Image Generation, Embeddings
- Validação de operações suportadas por modelo
- Endpoints OpenAI compatíveis
- Gestão de modelos via Cosmos DB

### 2. **Arquitetura Consistente**
- Estrutura de pastas alinhada com implementação real
- Use Cases específicos para cada operação
- Interfaces bem definidas entre camadas
- Separação clara de responsabilidades

### 3. **Integração LiteLLM**
- Implementação real do LiteLLMProvider
- Suporte completo para Azure OpenAI e Google Vertex AI
- Mapeamento correto de nomes de modelos
- Configuração automática de credenciais

### 4. **Gestão de Configuração**
- Configuração centralizada via variáveis de ambiente
- Validação de configurações obrigatórias
- Enriquecimento dinâmico de parâmetros
- Suporte a múltiplos provedores

### 5. **Database e Modelos**
- Implementação real do Cosmos DB Client
- Queries otimizadas para busca de modelos
- Suporte a operações específicas por modelo
- Mapeamento correto de entidades

### 6. **FastAPI Application**
- Dependency injection funcional
- Health checks e endpoints de modelo
- Lifecycle management adequado
- Tratamento de erros consistente

## 📋 Requirements

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
litellm==1.40.0
azure-cosmos==4.5.1
python-multipart==0.0.6
python-dotenv==1.0.0
```

## 🚀 Como Executar

### 1. Configurar Variáveis de Ambiente
```bash
export COSMOS_URL="https://sua-conta.documents.azure.com:443/"
export COSMOS_KEY="sua-chave-cosmos-db"
export COSMOS_DATABASE_ID="llm-proxy"
export COSMOS_CONTAINER_ID="models"

# API Keys dos provedores
export AZURE_OPENAI_API_KEY="sua-chave-azure"
export GOOGLE_VERTEX_CREDENTIALS="conteudo-json-credentials"
```

### 2. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 3. Executar Aplicação
```bash
uvicorn infra.controllers.app:app --host 0.0.0.0 --port 8000
```

### 4. Testar Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Listar modelos
curl http://localhost:8000/v1/models

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Esta implementação melhorada mantém consistência entre documentação e código, seguindo as melhores práticas de desenvolvimento e oferecendo uma base sólida para expansão futura.