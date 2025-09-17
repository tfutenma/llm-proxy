# LLM Proxy Microsserviço

## 📋 Visão Geral

Um microsserviço robusto em Python 3.11 usando FastAPI que atua como um proxy unificado para múltiplos provedores de LLM (Large Language Models). O serviço replica fielmente a interface da API da OpenAI, permitindo integração transparente com diversos provedores como Azure OpenAI, Google Vertex AI, Anthropic Claude e outros através da biblioteca LiteLLM.

### 🎯 Principais Benefícios

- **Interface Unificada**: Uma única API para acessar múltiplos provedores de LLM
- **Compatibilidade OpenAI**: Drop-in replacement para aplicações que já usam a API OpenAI
- **Gestão Centralizada**: Controle de modelos, permissões e configurações em um único lugar
- **Multi-Provider**: Suporte nativo para Azure, Google, Anthropic, OpenAI e outros
- **Segurança**: Gestão segura de API keys e credenciais
- **Escalabilidade**: Arquitetura preparada para alta demanda

## 🏗️ Arquitetura

O projeto segue uma **arquitetura hexagonal** (Ports and Adapters) que promove:
- **Separação de responsabilidades**: Código organizado e manutenível
- **Testabilidade**: Facilita testes unitários e de integração
- **Flexibilidade**: Fácil adição de novos provedores ou funcionalidades
- **Independência**: Domínio independente de frameworks e bibliotecas externas

### Camadas da Aplicação

| Camada | Responsabilidade | Componentes |
|--------|-----------------|-------------|
| **Application** | Coordena casos de uso e define interfaces | Interfaces, Use Cases |
| **Controllers** | Gerencia requisições HTTP e validações | Endpoints, DTOs, Routers |
| **Domain** | Lógica de negócio e regras do domínio | Entities, Services |
| **Infrastructure** | Implementações concretas e integrações | Cosmos DB, LiteLLM, Config |

## 📁 Estrutura de Diretórios

```bash
# Visualização completa da estrutura do projeto
llm-proxy/
├── 📂 llmproxy/                 # Pacote principal do projeto
│   ├── __init__.py
│   └── config.py               # Configurações centralizadas
├── 📂 application/              # Camada de aplicação
│   ├── 📂 interfaces/          # Contratos e abstrações
│   │   ├── __init__.py
│   │   ├── llm_provider_interface.py    # Interface para provedores LLM
│   │   └── llm_service_interface.py     # Interface para serviços LLM
│   └── 📂 use_cases/           # Casos de uso da aplicação
│       ├── __init__.py
│       ├── chat_completion_use_case.py  # UC: Chat/Completions
│       ├── text_to_speech_use_case.py   # UC: Text-to-Speech
│       ├── speech_to_text_use_case.py   # UC: Speech-to-Text
│       └── image_generation_use_case.py # UC: Geração de imagens
├── 📂 domain/                   # Camada de domínio
│   ├── 📂 entities/            # Entidades do domínio
│   │   ├── __init__.py
│   │   ├── llm_model.py        # Entidade LLMModel
│   │   └── operation.py        # Entidade Operation
│   └── 📂 services/            # Serviços de domínio
│       ├── __init__.py
│       ├── llm_model_service.py     # Serviço de modelos LLM
│       └── operation_validator.py   # Validador de operações
├── 📂 infra/                    # Camada de infraestrutura
│   ├── __init__.py
│   ├── 📂 controllers/         # Controladores HTTP
│   │   ├── __init__.py
│   │   ├── chat_completion_controller.py   # Controller de chat
│   │   ├── text_to_speech_controller.py    # Controller de TTS
│   │   ├── speech_to_text_controller.py    # Controller de STT
│   │   ├── image_generation_controller.py  # Controller de imagens
│   │   ├── model_controller.py             # Controller de modelos
│   │   └── app.py                         # Aplicação FastAPI principal
│   ├── 📂 schemas/             # Schemas de dados (DTOs)
│   │   ├── __init__.py
│   │   ├── chat_completion_schemas.py # Schemas para chat
│   │   ├── text_to_speech_schemas.py  # Schemas para TTS
│   │   ├── speech_to_text_schemas.py  # Schemas para STT
│   │   ├── image_generation_schemas.py # Schemas para imagens
│   │   └── model_schemas.py           # Schemas para modelos
│   ├── 📂 database/            # Integração com banco de dados
│   │   ├── __init__.py
│   │   ├── cosmos_client.py         # Cliente Cosmos DB
│   │   ├── model_repository.py      # Repositório de modelos
│   │   └── cosmos_models.py         # Modelos de dados Cosmos
│   └── 📂 providers/           # Integração com provedores LLM
│       ├── __init__.py
│       ├── litellm_provider.py      # Provider LiteLLM
│       ├── llm_adapters.py          # Adaptadores para diferentes LLMs
│       └── provider_utils.py        # Utilitários para provedores
├── 📄 requirements.txt          # Dependências Python
├── 🐳 docker-compose.yml        # Orquestração Docker
├── 🐳 Dockerfile               # Imagem Docker
├── 📄 .env.example             # Exemplo de variáveis de ambiente
└── 📄 README.md                # Este arquivo
```

## 🚀 Guia de Instalação Passo a Passo

### Pré-requisitos

- **Python 3.11+** instalado
- **Docker e Docker Compose** (opcional, para deployment containerizado)
- **Conta Azure** com Cosmos DB configurado
- **API Keys** dos provedores que deseja usar (OpenAI, Azure, Google, etc.)

### 1️⃣ Clone o Repositório

```bash
# Clone o projeto
git clone https://github.com/seu-usuario/llm-proxy.git
cd llm-proxy

# Crie um ambiente virtual Python
python -m venv venv

# Ative o ambiente virtual
# Linux/Mac:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate
```

### 2️⃣ Configure o Azure Cosmos DB

1. **Crie uma conta Cosmos DB no Azure Portal:**
   ```
   Portal Azure → Create Resource → Azure Cosmos DB → Core (SQL)
   ```

2. **Configure o banco de dados:**
   ```sql
   Database ID: llm-proxy
   Container ID: models
   Partition Key: /provider
   ```

3. **Obtenha as credenciais:**
   - No Azure Portal, vá para sua conta Cosmos DB
   - Settings → Keys
   - Copie a URI e Primary Key

### 3️⃣ Configure as Variáveis de Ambiente

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite o arquivo .env com suas configurações
nano .env  # ou use seu editor preferido
```

**Conteúdo do .env:**
```env
# === CONFIGURAÇÕES OBRIGATÓRIAS ===
# Cosmos DB (obtidas no passo anterior)
COSMOS_URL=https://sua-conta.documents.azure.com:443/
COSMOS_KEY=sua-chave-primaria-aqui
COSMOS_DATABASE_ID=llm-proxy
COSMOS_CONTAINER_ID=models

# Ambiente
ENVIRONMENT=development  # ou production
HOST=0.0.0.0
PORT=8000

# === API KEYS DOS PROVEDORES ===
# Configure apenas os que você vai usar

# OpenAI
OPENAI_API_KEY=sk-...

# Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://seu-recurso.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01

# Google Vertex AI
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
GOOGLE_PROJECT_ID=seu-projeto-id

# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-...
```

### 4️⃣ Instale as Dependências

```bash
# Instale todas as dependências
pip install -r requirements.txt

# Verifique a instalação
python -c "import fastapi; print(f'FastAPI {fastapi.__version__} instalado com sucesso!')"
```

### 5️⃣ Configure os Modelos no Cosmos DB

Crie documentos no container `models` do Cosmos DB seguindo a estrutura padrão:

**Campos obrigatórios:**
- `id` - Identificador único do modelo
- `projects` - Lista com IDs dos projetos (para modelos privados)
- `private` - Boolean indicando se é privado
- `provider` - String: "Azure" ou "Google"
- `model_type` - Tipo específico (ver tabela abaixo)
- `name` - Nome descritivo do modelo
- `parameters` - Dicionário com configurações
- `costs` - Dicionário com custos de entrada/saída
- `operations` - Lista de operações permitidas

```json
// Exemplo 1: GPT-4 via Azure OpenAI
{
  "id": "gpt-4-azure",
  "name": "GPT-4 Azure",
  "code": "gpt-4",
  "projects": [],
  "private": false,
  "provider": "Azure",
  "model_type": "AzureOpenAI",
  "parameters": {
    "secret_name": "AZURE_OPENAI_API_KEY",
    "deployment_name": "gpt-4-deployment",
    "api_version": "2024-02-01",
    "endpoint": "https://meu-recurso.openai.azure.com/",
    "enable_tools": true
  },
  "costs": {
    "currency": "USD",
    "cost_input_1Mtokens": 0.03,
    "cost_output_1Mtokens": 0.06
  },
  "operations": ["ChatCompletion", "Responses", "embeddings"]
}

// Exemplo 2: DeepSeek via Google Vertex AI
{
  "id": "deepseek-vertex",
  "name": "DeepSeek V2 via Vertex AI",
  "code": "deepseek-v2",
  "projects": [],
  "private": false,
  "provider": "Google",
  "model_type": "GoogleVertexDeepSeek",
  "parameters": {
    "secret_name": "SP-BRIDGE-4852-DEV",
    "gcp_project": "meu-projeto-gcp",
    "location": "us-central1",
    "enable_tools": true
  },
  "costs": {
    "input": 0.014,
    "output": 0.028
  },
  "operations": ["ChatCompletion", "Responses"]
}

// Exemplo 3: Claude via Google Vertex AI
{
  "id": "claude-vertex",
  "name": "Claude 3 Sonnet via Vertex AI",
  "code": "claude-3-sonnet",
  "projects": [],
  "private": false,
  "provider": "Google",
  "model_type": "GoogleVertexClaude",
  "parameters": {
    "secret_name": "SP-BRIDGE-4852-DEV",
    "gcp_project": "meu-projeto-gcp",
    "location": "us-central1",
    "enable_tools": true
  },
  "costs": {
    "input": 0.003,
    "output": 0.015
  },
  "operations": ["ChatCompletion", "Responses", "images"]
}
```

**⚠️ Importantes:**

**Parameters (Azure):**
- `secret_name` - Variável de ambiente com credenciais Azure
- `deployment_name` - Nome do deployment no Azure
- `api_version` - Versão da API (ex: "2024-02-01")
- `endpoint` - Endpoint do Azure OpenAI
- `enable_tools` - Boolean para habilitar tools

**Parameters (Google):**
- `secret_name` - Variável de ambiente com credenciais Google (ex: "SP-BRIDGE-4852-DEV")
- `gcp_project` - ID do projeto GCP
- `location` - Localização (ex: "us-central1")
- `enable_tools` - Boolean para habilitar tools

**Costs:**
- `input` - Custo por 1k tokens de entrada (USD)
- `output` - Custo por 1k tokens de saída (USD)

**Operations:**
- `Responses` - Geração de respostas básicas
- `ChatCompletion` - Chat completions
- `embeddings` - Geração de embeddings
- `images` - Geração/análise de imagens
- `audio_tts` - Text-to-Speech
- `audio_stt` - Speech-to-Text
- `audio_realtime` - Áudio em tempo real

### 6️⃣ Execute o Serviço

#### Opção A: Execução Local

```bash
# Modo desenvolvimento (com hot reload)
uvicorn infra.controllers.main:app --reload --host 0.0.0.0 --port 8000

# Modo produção
uvicorn infra.controllers.main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Opção B: Usando Docker

```bash
# Build e execução com docker-compose
docker-compose up --build

# Ou executar em background
docker-compose up -d --build

# Ver logs
docker-compose logs -f llm-proxy

# Parar o serviço
docker-compose down
```

### 7️⃣ Verifique a Instalação

```bash
# Health check
curl http://localhost:8000/health

# Listar modelos disponíveis
curl http://localhost:8000/v1/models

# Teste de chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Olá, teste!"}
    ]
  }'
```

## 💻 Implementação Completa dos Arquivos

application/interfaces/__init__.py
```python
# Interfaces da camada de aplicação
```


application/interfaces/llm_provider_interface.py
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
        """Retorna o nome do provedor (azure, google, openai, anthropic, etc)"""
        pass

    @abstractmethod
    def supports_operation(self, operation: str) -> bool:
        """Verifica se o provedor suporta uma operação específica"""
        pass
```


application/interfaces/llm_service_interface.py
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


application/use_cases/__init__.py
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


application/use_cases/chat_completion_use_case.py
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

application/use_cases/text_to_speech_use_case.py
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

application/use_cases/speech_to_text_use_case.py
```python
from typing import Dict, Any
from application.interfaces.llm_service_interface import LLMServiceInterface
from infra.schemas.speech_to_text_schemas import SpeechToTextRequest
from infra.providers.litellm_provider import LiteLLMProvider

class SpeechToTextUseCase:
    """Caso de uso para speech-to-text com operações assíncronas"""

    def __init__(self, model_service: LLMServiceInterface):
        self.model_service = model_service

    async def execute(self, request: SpeechToTextRequest) -> Dict[Any, Any]:
        """Executa uma requisição de speech-to-text de forma assíncrona"""
        # Obtém configuração do modelo
        model = await self.model_service.get_model(request.model)
        if not model:
            raise ValueError(f"Model '{request.model}' not found")

        # Verifica se o modelo suporta STT
        if not await self.model_service.validate_model_for_operation(request.model, 'audio_stt'):
            raise ValueError(f"Model '{request.model}' does not support speech-to-text")

        # Cria instância do provedor e executa requisição
        provider = LiteLLMProvider(model)

        # Converte requisição para formato do provedor
        stt_params = {
            'file': request.file,
            'language': request.language,
            'prompt': request.prompt,
            'response_format': request.response_format,
            'temperature': request.temperature
        }

        # Remove valores None
        stt_params = {k: v for k, v in stt_params.items() if v is not None}

        return await provider.speech_to_text(**stt_params)
```

application/use_cases/image_generation_use_case.py
```python
from typing import Dict, Any
from application.interfaces.llm_service_interface import LLMServiceInterface
from infra.schemas.image_generation_schemas import ImageGenerationRequest
from infra.providers.litellm_provider import LiteLLMProvider

class ImageGenerationUseCase:
    """Caso de uso para geração de imagens com operações assíncronas"""

    def __init__(self, model_service: LLMServiceInterface):
        self.model_service = model_service

    async def execute(self, request: ImageGenerationRequest) -> Dict[Any, Any]:
        """Executa uma requisição de geração de imagem de forma assíncrona"""
        # Obtém configuração do modelo
        model = await self.model_service.get_model(request.model)
        if not model:
            raise ValueError(f"Model '{request.model}' not found")

        # Verifica se o modelo suporta geração de imagens
        if not await self.model_service.validate_model_for_operation(request.model, 'images'):
            raise ValueError(f"Model '{request.model}' does not support image generation")

        # Cria instância do provedor e executa requisição
        provider = LiteLLMProvider(model)

        # Converte requisição para formato do provedor
        image_params = {
            'prompt': request.prompt,
            'n': request.n,
            'size': request.size,
            'response_format': request.response_format,
            'user': request.user
        }

        # Remove valores None
        image_params = {k: v for k, v in image_params.items() if v is not None}

        return await provider.generate_image(**image_params)
```


infra/dto/__init__.py
```
# Empty file to make it a package
```


infra/dto/chat_completions.py
```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionsRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    user: Optional[str] = None

class ChatCompletionsResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
```

infra/dto/tts.py
```python
from pydantic import BaseModel, Field
from typing import Optional

class TTSRequest(BaseModel):
    model: str
    input: str = Field(..., max_length=4096)
    voice: str = Field(default="alloy")
    response_format: Optional[str] = Field(default="mp3")
    speed: Optional[float] = Field(default=1.0, ge=0.25, le=4.0)

class TTSResponse(BaseModel):
    audio: bytes
    content_type: str = "audio/mpeg"
```


infra/dto/stt.py
```python
from pydantic import BaseModel, Field
from typing import Optional
from fastapi import UploadFile

class STTRequest(BaseModel):
    file: UploadFile = Field(...)
    model: str
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = Field(default="json")
    temperature: Optional[float] = Field(default=0, ge=0.0, le=1.0)

    class Config:
        arbitrary_types_allowed = True

class STTResponse(BaseModel):
    text: str
```


infra/dto/image.py
```python
from pydantic import BaseModel, Field
from typing import Optional

class ImageRequest(BaseModel):
    model: str
    prompt: str = Field(..., max_length=1000)
    n: Optional[int] = Field(default=1, ge=1, le=10)
    size: Optional[str] = Field(default="1024x1024")
    response_format: Optional[str] = Field(default="url")
    user: Optional[str] = None

class ImageResponse(BaseModel):
    created: int
    data: list
```

infra/controllers/__init__.py
```
# Empty file to make it a package
```


infra/controllers/chat_completions_controller.py
```python
from fastapi import APIRouter, HTTPException, Depends
from infra.dto.chat_completions import ChatCompletionsRequest
from application.use_cases.chat_completions import ChatCompletionsUseCase
from domain.services.llm_model_service import LLMModelService

router = APIRouter()

@router.post("/chat/completions", response_model=dict)
async def chat_completions(
    request: ChatCompletionsRequest,
    service: LLMModelService = Depends()
):
    """OpenAI-compatible chat completions endpoint"""
    try:
        use_case = ChatCompletionsUseCase(service)
        response = await use_case.execute(request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
```


infra/controllers/tts_controller.py
```python
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response
from infra.dto.tts import TTSRequest
from application.use_cases.tts import TTSUseCase
from domain.services.llm_model_service import LLMModelService

router = APIRouter()

@router.post("/audio/speech")
async def text_to_speech(
    request: TTSRequest,
    service: LLMModelService = Depends()
):
    """OpenAI-compatible text-to-speech endpoint"""
    try:
        use_case = TTSUseCase(service)
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


infra/controllers/stt_controller.py
```python
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form
from infra.dto.stt import STTResponse
from application.use_cases.stt import STTUseCase
from domain.services.llm_model_service import LLMModelService

router = APIRouter()

@router.post("/audio/transcriptions", response_model=STTResponse)
async def speech_to_text(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str = Form(None),
    prompt: str = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    service: LLMModelService = Depends()
):
    """OpenAI-compatible speech-to-text endpoint"""
    try:
        # Create request object manually since we're dealing with multipart form
        from infra.dto.stt import STTRequest
        request = STTRequest(
            file=file,
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature
        )

        use_case = STTUseCase(service)
        response = await use_case.execute(request)

        return STTResponse(text=response.get('text', ''))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
```


infra/controllers/image_controller.py
```python
from fastapi import APIRouter, HTTPException, Depends
from infra.dto.image import ImageRequest, ImageResponse
from application.use_cases.image import ImageUseCase
from domain.services.llm_model_service import LLMModelService
import time

router = APIRouter()

@router.post("/images/generations", response_model=ImageResponse)
async def generate_image(
    request: ImageRequest,
    service: LLMModelService = Depends()
):
    """OpenAI-compatible image generation endpoint"""
    try:
        use_case = ImageUseCase(service)
        response = await use_case.execute(request)

        return ImageResponse(
            created=int(time.time()),
            data=response.get('data', [])
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
```


infra/controllers/main.py
```python
from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager
from llmproxy.config import Config
from infra.cosmos.client import CosmosDBClient
from domain.services.llm_model_service import LLMModelService
from infra.controllers.chat_completions_controller import router as chat_router
from infra.controllers.tts_controller import router as tts_router
from infra.controllers.stt_controller import router as stt_router
from infra.controllers.image_controller import router as image_router

# Global variables for dependency injection
db_client = None
model_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global db_client, model_service

    # Validate required configurations
    try:
        Config.validate_required_configs()
        print("Configuration validation passed")
    except ValueError as e:
        print(f"Configuration error: {e}")
        raise

    # Initialize database client and services
    db_client = CosmosDBClient()
    model_service = LLMModelService(db_client)

    print("Application startup completed")
    yield

    # Shutdown
    print("Application shutdown")

app = FastAPI(
    title="LLM Proxy Service",
    description="A proxy service for multiple LLM providers",
    version="1.0.0",
    lifespan=lifespan
)

# Dependency to get model service
def get_model_service() -> LLMModelService:
    if model_service is None:
        raise HTTPException(status_code=500, detail="Model service not initialized")
    return model_service

# Include routers
app.include_router(
    chat_router,
    prefix="/v1",
    dependencies=[Depends(get_model_service)],
    tags=["chat"]
)

app.include_router(
    tts_router,
    prefix="/v1",
    dependencies=[Depends(get_model_service)],
    tags=["audio"]
)

app.include_router(
    stt_router,
    prefix="/v1",
    dependencies=[Depends(get_model_service)],
    tags=["audio"]
)

app.include_router(
    image_router,
    prefix="/v1",
    dependencies=[Depends(get_model_service)],
    tags=["images"]
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "environment": Config.ENVIRONMENT}

@app.get("/v1/models")
async def list_models(service: LLMModelService = Depends(get_model_service)):
    """List available models - OpenAI compatible"""
    models = service.list_models()
    return {
        "object": "list",
        "data": [
            {
                "id": model.code,
                "object": "model",
                "created": 1677610602,
                "owned_by": model.provider,
                "permission": [],
                "root": model.code,
                "parent": None
            }
            for model in models
        ]
    }
```


domain/entities/init.py
```
# Empty file to make it a package
```


domain/entities/model.py
```python
from typing import List, Dict, Any

class LLMModel:
    def __init__(
        self,
        id: str,
        projects: List[str],
        private: bool,
        provider: str,  # Azure, Google
        model_type: str,  # AzureOpenAI, GoogleVertexDeepSeek, etc.
        name: str,
        parameters: Dict[str, Any],  # secret_name, deployment_name, location, gcp_project, enable_tools, api_version, endpoint
        costs: Dict[str, float],  # custos de entrada e saída
        operations: List[str],  # operações permitidas
        **kwargs  # Para campos adicionais do Cosmos
    ):
        self.id = id
        self.projects = projects
        self.private = private
        self.provider = provider
        self.model_type = model_type
        self.name = name
        self.parameters = parameters
        self.costs = costs
        self.operations = operations

        # Campos adicionais do Cosmos DB
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"LLMModel(id={self.id}, name={self.name}, provider={self.provider}, type={self.model_type})"

    def get_provider_config(self) -> Dict[str, Any]:
        """Retorna configuração específica do provider"""
        provider_configs = {
            'Azure': ['secret_name', 'deployment_name', 'api_version', 'endpoint'],
            'Google': ['secret_name', 'gcp_project', 'location', 'enable_tools']
        }

        config = {}
        if self.provider in provider_configs:
            for key in provider_configs[self.provider]:
                if key in self.parameters:
                    config[key] = self.parameters[key]
        return config

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calcula o custo baseado nos tokens de entrada e saída"""
        input_cost = input_tokens * self.costs.get('input', 0.0) / 1000  # custo por 1k tokens
        output_cost = output_tokens * self.costs.get('output', 0.0) / 1000
        return input_cost + output_cost

    def supports_operation(self, operation: str) -> bool:
        """Verifica se o modelo suporta uma operação específica"""
        return operation in self.operations

    def get_cost_info(self) -> Dict[str, float]:
        """Retorna informações de custo do modelo"""
        return {
            'input_cost_per_1k': self.costs.get('input', 0.0),
            'output_cost_per_1k': self.costs.get('output', 0.0)
        }

    def get_supported_operations(self) -> List[str]:
        """Retorna lista de operações suportadas"""
        return self.operations.copy()

    def is_available_for_project(self, project_id: str) -> bool:
        """Check if the model is available for a specific project"""
        if not self.private:
            return True
        return project_id in self.projects

    def supports_streaming(self) -> bool:
        """Check if the model supports streaming responses"""
        return self.supports_operation('ChatCompletion')
```

# Arquivo removido - Provider agora é apenas um campo string no modelo
# O provider é cadastrado diretamente no documento do modelo no Cosmos DB

domain/services/init.py
```
# Empty file to make it a package
```


domain/services/llm_model_service.py
```python
from typing import List, Optional
from domain.entities.model import LLMModel
from infra.cosmos.client import CosmosDBClient
from llmproxy.config import Config

class LLMModelService:
    def __init__(self, db_client: CosmosDBClient):
        self.db_client = db_client

    def get_model(self, model_id: str) -> Optional[LLMModel]:
        """Get a specific model by ID"""
        try:
            model_data = self.db_client.get_model(model_id)
            if model_data:
                # Enrich model with environment variables for parameters
                enriched_parameters = self._enrich_parameters(model_data.get('parameters', {}))
                model_data['parameters'] = enriched_parameters
                return LLMModel(**model_data)
            return None
        except Exception as e:
            print(f"Error getting model {model_id}: {str(e)}")
            return None

    def list_models(self) -> List[LLMModel]:
        """List all available models"""
        try:
            models_data = self.db_client.list_models()
            models = []
            for model_data in models_data:
                enriched_parameters = self._enrich_parameters(model_data.get('parameters', {}))
                model_data['parameters'] = enriched_parameters
                models.append(LLMModel(**model_data))
            return models
        except Exception as e:
            print(f"Error listing models: {str(e)}")
            return []

    def _enrich_parameters(self, parameters: dict) -> dict:
        """Enrich parameters with values from environment variables"""
        enriched = {}
        for key, env_var_name in parameters.items():
            if isinstance(env_var_name, str) and env_var_name.isupper():
                # If the value is an uppercase string, treat it as an environment variable name
                env_value = Config.get_provider_config(env_var_name)
                if env_value:
                    enriched[key] = env_value
                else:
                    print(f"Warning: Environment variable '{env_var_name}' not found for parameter '{key}'")
                    enriched[key] = env_var_name  # Keep original value as fallback
            else:
                # If it's not a string or not uppercase, use the value as is
                enriched[key] = env_var_name
        return enriched

    def get_models_by_provider(self, provider: str) -> List[LLMModel]:
        """Get models filtered by provider"""
        try:
            models_data = self.db_client.get_models_by_provider(provider)
            models = []
            for model_data in models_data:
                enriched_parameters = self._enrich_parameters(model_data.get('parameters', {}))
                model_data['parameters'] = enriched_parameters
                models.append(LLMModel(**model_data))
            return models
        except Exception as e:
            print(f"Error getting models for provider {provider}: {str(e)}")
            return []

    def get_models_by_type(self, model_type: str) -> List[LLMModel]:
        """Get models filtered by type"""
        try:
            models_data = self.db_client.get_models_by_type(model_type)
            models = []
            for model_data in models_data:
                enriched_parameters = self._enrich_parameters(model_data.get('parameters', {}))
                model_data['parameters'] = enriched_parameters
                models.append(LLMModel(**model_data))
            return models
        except Exception as e:
            print(f"Error getting models for type {model_type}: {str(e)}")
            return []

    def get_models_for_project(self, project_id: str) -> List[LLMModel]:
        """Get models available for a specific project"""
        all_models = self.list_models()
        return [model for model in all_models if model.is_available_for_project(project_id)]

    def get_models_by_operation(self, operation: str) -> List[LLMModel]:
        """Get models filtered by supported operation"""
        all_models = self.list_models()
        return [model for model in all_models if model.supports_operation(operation)]
```


llmproxy/__init__.py
```
# Pacote principal do projeto LLM Proxy
```


llmproxy/config.py
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

    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    @classmethod
    def get_provider_config(cls, secret_name: str) -> Optional[str]:
        """Get provider configuration from environment variables"""
        return os.getenv(secret_name)

    @classmethod
    def get_google_credentials(cls, secret_name: str) -> Optional[str]:
        """Get Google credentials from environment variables

        Args:
            secret_name: Nome da variável de ambiente (ex: SP-BRIDGE-4852-DEV)

        Returns:
            Conteúdo das credenciais Google ou None se não encontrado
        """
        credentials_content = os.getenv(secret_name)
        if not credentials_content:
            print(f"Warning: Google credentials not found for secret_name: {secret_name}")
            return None
        return credentials_content

    @classmethod
    def validate_required_configs(cls):
        """Validate that all required configurations are present"""
        required_configs = [
            ("COSMOS_URL", cls.COSMOS_URL),
            ("COSMOS_KEY", cls.COSMOS_KEY),
        ]

        missing_configs = [name for name, value in required_configs if not value]

        if missing_configs:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_configs)}")
```


infra/cosmos/__init__.py
```
# Módulo de integração com Azure Cosmos DB
```


infra/cosmos/client.py
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

        except exceptions.CosmosHttpResponseError as e:
            print(f"Error querying model {model_id}: {str(e)}")
            return None

    def list_models(self) -> List[Dict[Any, Any]]:
        """List all models"""
        try:
            query = "SELECT * FROM c"
            items = list(self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            return items

        except exceptions.CosmosHttpResponseError as e:
            print(f"Error listing models: {str(e)}")
            return []

    def get_models_by_provider(self, provider: str) -> List[Dict[Any, Any]]:
        """Get models filtered by provider"""
        try:
            query = "SELECT * FROM c WHERE c.provider = @provider"
            parameters = [{"name": "@provider", "value": provider}]

            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            return items

        except exceptions.CosmosHttpResponseError as e:
            print(f"Error querying models by provider {provider}: {str(e)}")
            return []

    def get_models_by_type(self, model_type: str) -> List[Dict[Any, Any]]:
        """Get models filtered by type"""
        try:
            query = "SELECT * FROM c WHERE c.model_type = @model_type"
            parameters = [{"name": "@model_type", "value": model_type}]

            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            return items

        except exceptions.CosmosHttpResponseError as e:
            print(f"Error querying models by type {model_type}: {str(e)}")
            return []

    def get_models_by_project(self, project_id: str) -> List[Dict[Any, Any]]:
        """Get models available for a specific project"""
        try:
            query = "SELECT * FROM c WHERE c.private = false OR ARRAY_CONTAINS(c.projects, @project_id)"
            parameters = [{"name": "@project_id", "value": project_id}]

            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            return items

        except exceptions.CosmosHttpResponseError as e:
            print(f"Error querying models by project {project_id}: {str(e)}")
            return []

    def get_models_by_operation(self, operation: str) -> List[Dict[Any, Any]]:
        """Get models that support a specific operation"""
        try:
            query = "SELECT * FROM c WHERE ARRAY_CONTAINS(c.operations, @operation)"
            parameters = [{"name": "@operation", "value": operation}]

            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            return items

        except exceptions.CosmosHttpResponseError as e:
            print(f"Error querying models by operation {operation}: {str(e)}")
            return []
```


infra/litellm/__init__.py
```
# Módulo de integração com LiteLLM
```


infra/litellm/provider.py
```python
import litellm
from typing import Dict, Any, Optional
from application.interfaces.model import LLMInterface
from domain.entities.model import LLMModel
import base64
import io

class LiteLLMProvider(LLMInterface):
    def __init__(self, model: LLMModel):
        self.model = model
        self.parameters = model.parameters

        # Set up LiteLLM with the model parameters
        self._setup_litellm()

    def _setup_litellm(self):
        """Setup LiteLLM with the model parameters"""
        # Set global API keys for LiteLLM
        if 'api_key' in self.parameters:
            litellm.api_key = self.parameters['api_key']

        if 'api_base' in self.parameters:
            litellm.api_base = self.parameters['api_base']

        if 'api_version' in self.parameters:
            litellm.api_version = self.parameters['api_version']

        # Setup Google Vertex AI credentials
        if self._is_google_vertex_model():
            self._setup_google_credentials()

    def chat_completions(self, **kwargs) -> Dict[Any, Any]:
        """Handle chat completions using LiteLLM"""
        try:
            # Prepare the model name for LiteLLM
            model_name = self._prepare_model_name()

            # Override model in kwargs
            kwargs['model'] = model_name

            # Add any additional parameters specific to this model
            additional_params = self._get_additional_parameters()
            kwargs.update(additional_params)

            response = litellm.completion(**kwargs)

            # Convert to OpenAI format if needed
            return self._format_chat_response(response)

        except Exception as e:
            print(f"Error in chat completions: {str(e)}")
            raise

    def tts(self, **kwargs) -> Dict[Any, Any]:
        """Handle text-to-speech using LiteLLM"""
        try:
            # Prepare the model name for LiteLLM TTS
            model_name = self._prepare_model_name()
            kwargs['model'] = model_name

            # Add TTS specific parameters
            kwargs.update(self._get_additional_parameters())

            # For now, this is a placeholder as LiteLLM's TTS support varies
            # You might need to implement provider-specific logic
            if self.model.provider.lower() == 'openai':
                response = litellm.speech(**kwargs)
            else:
                # Implement provider-specific TTS logic here
                raise NotImplementedError(f"TTS not implemented for provider: {self.model.provider}")

            return {'audio': response}

        except Exception as e:
            print(f"Error in TTS: {str(e)}")
            raise

    def stt(self, **kwargs) -> Dict[Any, Any]:
        """Handle speech-to-text using LiteLLM"""
        try:
            # Prepare the model name for LiteLLM STT
            model_name = self._prepare_model_name()
            kwargs['model'] = model_name

            # Add STT specific parameters
            kwargs.update(self._get_additional_parameters())

            # For now, this is a placeholder as LiteLLM's STT support varies
            if self.model.provider.lower() == 'openai':
                response = litellm.transcription(**kwargs)
            else:
                # Implement provider-specific STT logic here
                raise NotImplementedError(f"STT not implemented for provider: {self.model.provider}")

            return response

        except Exception as e:
            print(f"Error in STT: {str(e)}")
            raise

    def generate_image(self, **kwargs) -> Dict[Any, Any]:
        """Handle image generation using LiteLLM"""
        try:
            # Prepare the model name for LiteLLM image generation
            model_name = self._prepare_model_name()
            kwargs['model'] = model_name

            # Add image generation specific parameters
            kwargs.update(self._get_additional_parameters())

            response = litellm.image_generation(**kwargs)
            return response

        except Exception as e:
            print(f"Error in image generation: {str(e)}")
            raise

    def _prepare_model_name(self) -> str:
        """Prepare the model name for LiteLLM based on provider and model_type"""
        provider = self.model.provider.lower()
        model_type = self.model.model_type

        # Handle Azure models
        if model_type == 'AzureOpenAI' or provider == 'azure':
            deployment_name = self.parameters.get('deployment_name', self.model.code)
            return f"azure/{deployment_name}"
        
        elif model_type == 'AzureFoundry':
            deployment_name = self.parameters.get('deployment_name', self.model.code)
            return f"azure/{deployment_name}"
        
        # Handle Google Vertex AI models
        elif model_type == 'GoogleVertex':
            return f"vertex_ai/{self.model.code}"
        
        elif model_type == 'GoogleVertexClaude':
            return f"vertex_ai/claude-{self.model.code}"
        
        elif model_type == 'GoogleVertexDeepSeek':
            # DeepSeek no Vertex AI
            return f"vertex_ai/deepseek-{self.model.code}"
        
        elif model_type == 'GoogleVertexMeta':
            return f"vertex_ai/llama-{self.model.code}"
        
        elif model_type == 'GoogleVertexMistral':
            return f"vertex_ai/mistral-{self.model.code}"
        
        elif model_type == 'GoogleVertexJamba':
            return f"vertex_ai/jamba-{self.model.code}"
        
        elif model_type == 'GoogleVertexQwen':
            return f"vertex_ai/qwen-{self.model.code}"
        
        elif model_type == 'GoogleVertexOpenAI':
            return f"vertex_ai/gpt-{self.model.code}"
        
        # Handle direct providers
        elif provider == 'openai':
            return self.model.code
        
        elif provider == 'anthropic':
            return f"claude-{self.model.code}"
        
        else:
            # Default to the model code
            return self.model.code

    def _get_additional_parameters(self) -> Dict[str, Any]:
        """Get additional parameters that should be passed to LiteLLM"""
        additional_params = {}

        # Map common parameters
        param_mapping = {
            'temperature': float,
            'max_tokens': int,
            'top_p': float,
            'frequency_penalty': float,
            'presence_penalty': float,
            'timeout': float,
            'api_base': str,
            'api_version': str,
            'api_key': str
        }
        
        # Add Google Vertex AI specific parameters
        if self._is_google_vertex_model():
            if 'project_id' in self.parameters:
                additional_params['vertex_project'] = self.parameters['project_id']
            if 'location' in self.parameters:
                additional_params['vertex_location'] = self.parameters['location']
            
            # Add vertex_credentials if available
            secret_name = self.parameters.get('secret_name')
            if secret_name:
                from llmproxy.config import Config
                credentials_content = Config.get_google_credentials(secret_name)
                if credentials_content:
                    additional_params['vertex_credentials'] = credentials_content

        for param_name, param_type in param_mapping.items():
            if param_name in self.parameters:
                try:
                    additional_params[param_name] = param_type(self.parameters[param_name])
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert parameter {param_name} to {param_type}")

        return additional_params

    def _format_chat_response(self, response) -> Dict[Any, Any]:
        """Format the response to match OpenAI's format"""
        if hasattr(response, 'model_dump'):
            # If it's a Pydantic model
            return response.model_dump()
        elif hasattr(response, 'to_dict'):
            # If it has a to_dict method
            return response.to_dict()
        elif isinstance(response, dict):
            # If it's already a dict
            return response
        else:
            # Try to convert to dict
            try:
                return dict(response)
            except:
                return {"error": "Could not format response", "raw_response": str(response)}
    
    def _is_google_vertex_model(self) -> bool:
        """Check if the model is a Google Vertex AI model"""
        google_vertex_types = [
            'GoogleVertex', 'GoogleVertexClaude', 'GoogleVertexDeepSeek',
            'GoogleVertexMeta', 'GoogleVertexMistral', 'GoogleVertexJamba',
            'GoogleVertexQwen', 'GoogleVertexOpenAI'
        ]
        return self.model.model_type in google_vertex_types or self.model.provider.lower() in ['google', 'vertex_ai']
    
    def _setup_google_credentials(self):
        """Setup Google Vertex AI credentials"""
        try:
            from llmproxy.config import Config
            
            # Busca pelo secret_name nos parâmetros
            secret_name = self.parameters.get('secret_name')
            if secret_name:
                # Carrega credenciais da variável de ambiente
                credentials_content = Config.get_google_credentials(secret_name)
                if credentials_content:
                    # Passa as credenciais para o LiteLLM
                    import tempfile
                    import json
                    
                    # Cria arquivo temporário com as credenciais
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        f.write(credentials_content)
                        credentials_path = f.name
                    
                    # Define no ambiente para o LiteLLM
                    import os
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
                    
                    print(f"Google credentials loaded from {secret_name}")
                else:
                    print(f"Warning: Could not load Google credentials from {secret_name}")
            else:
                print("Warning: secret_name not found in model parameters for Google Vertex model")
        
        except Exception as e:
            print(f"Error setting up Google credentials: {str(e)}")
```


infra/cosmos/repositories.py
```python
from typing import List, Optional, Dict, Any
from infra.cosmos.client import CosmosDBClient
from domain.entities.model import LLMModel

class ModelRepository:
    """Repositório para operações com modelos no Cosmos DB"""

    def __init__(self, client: CosmosDBClient):
        self.client = client

    def find_by_id(self, model_id: str) -> Optional[LLMModel]:
        """Busca um modelo por ID"""
        data = self.client.get_model(model_id)
        return LLMModel(**data) if data else None

    def find_all(self) -> List[LLMModel]:
        """Lista todos os modelos"""
        models_data = self.client.list_models()
        return [LLMModel(**data) for data in models_data]

    def find_by_provider(self, provider: str) -> List[LLMModel]:
        """Busca modelos por provedor (provider é string)"""
        models_data = self.client.get_models_by_provider(provider)
        return [LLMModel(**data) for data in models_data]

    def list_providers(self) -> List[str]:
        """Lista todos os providers únicos (strings)"""
        all_models = self.find_all()
        return list(set(model.provider for model in all_models))

    def find_by_type(self, model_type: str) -> List[LLMModel]:
        """Busca modelos por tipo"""
        models_data = self.client.get_models_by_type(model_type)
        return [LLMModel(**data) for data in models_data]

    def find_by_project(self, project_id: str) -> List[LLMModel]:
        """Busca modelos disponíveis para um projeto"""
        models_data = self.client.get_models_by_project(project_id)
        return [LLMModel(**data) for data in models_data]

    def find_by_operation(self, operation: str) -> List[LLMModel]:
        """Busca modelos que suportam uma operação específica"""
        models_data = self.client.get_models_by_operation(operation)
        return [LLMModel(**data) for data in models_data]
```


infra/cosmos/models.py
```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class CosmosModelDocument(BaseModel):
    """Documento de modelo no Cosmos DB - Estrutura padrão"""
    id: str
    projects: List[str] = Field(default_factory=list)  # IDs dos projetos
    private: bool = False
    provider: str  # "Azure" ou "Google"
    model_type: str  # AzureOpenAI, GoogleVertexDeepSeek, etc.
    name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)  # secret_name, deployment_name, location, gcp_project, enable_tools, api_version, endpoint
    costs: Dict[str, float] = Field(default_factory=dict)  # input, output (custo por 1k tokens)
    operations: List[str] = Field(default_factory=list)  # operações permitidas

    # Campos automáticos do Cosmos DB
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    _rid: Optional[str] = None
    _self: Optional[str] = None
    _etag: Optional[str] = None
    _attachments: Optional[str] = None
    _ts: Optional[int] = None

    class Config:
        extra = "allow"
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def validate_operations(self) -> bool:
        """Valida se as operações são válidas"""
        valid_operations = {
            "Responses", "ChatCompletion", "embeddings",
            "images", "audio_tts", "audio_stt", "audio_realtime"
        }
        return all(op in valid_operations for op in self.operations)

    def validate_provider(self) -> bool:
        """Valida se o provider é válido"""
        return self.provider in ["Azure", "Google"]

    def validate_model_type(self) -> bool:
        """Valida se o model_type é válido"""
        valid_types = [
            "AzureOpenAI", "AzureFoundry",
            "GoogleVertex", "GoogleVertexClaude", "GoogleVertexDeepSeek",
            "GoogleVertexMeta", "GoogleVertexMistral", "GoogleVertexJamba",
            "GoogleVertexQwen", "GoogleVertexOpenAI"
        ]
        return self.model_type in valid_types

    def get_required_parameters(self) -> List[str]:
        """Retorna parâmetros obrigatórios baseado no provider"""
        if self.provider == "Azure":
            return ["secret_name", "deployment_name", "api_version", "endpoint"]
        elif self.provider == "Google":
            return ["secret_name", "gcp_project", "location"]
        return []
```


infra/litellm/adapters.py
```python
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import litellm

class LLMAdapter(ABC):
    """Classe base para adaptadores de diferentes provedores LLM"""

    @abstractmethod
    def prepare_request(self, **kwargs) -> Dict[str, Any]:
        """Prepara a requisição para o formato do provedor"""
        pass

    @abstractmethod
    def parse_response(self, response: Any) -> Dict[str, Any]:
        """Converte a resposta para o formato padrão OpenAI"""
        pass

class AzureOpenAIAdapter(LLMAdapter):
    """Adaptador para Azure OpenAI"""

    def __init__(self, deployment_name: str, api_base: str, api_version: str):
        self.deployment_name = deployment_name
        self.api_base = api_base
        self.api_version = api_version

    def prepare_request(self, **kwargs) -> Dict[str, Any]:
        kwargs['model'] = f"azure/{self.deployment_name}"
        kwargs['api_base'] = self.api_base
        kwargs['api_version'] = self.api_version
        return kwargs

    def parse_response(self, response: Any) -> Dict[str, Any]:
        if hasattr(response, 'model_dump'):
            return response.model_dump()
        return dict(response)

class AnthropicAdapter(LLMAdapter):
    """Adaptador para Anthropic Claude"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def prepare_request(self, **kwargs) -> Dict[str, Any]:
        kwargs['model'] = f"claude-{self.model_name}"
        # Adaptações específicas do Anthropic
        if 'max_tokens' in kwargs and kwargs['max_tokens'] is None:
            kwargs['max_tokens'] = 4096
        return kwargs

    def parse_response(self, response: Any) -> Dict[str, Any]:
        # Converte resposta Anthropic para formato OpenAI
        if hasattr(response, 'model_dump'):
            return response.model_dump()
        return dict(response)

class GoogleVertexAIAdapter(LLMAdapter):
    """Adaptador para Google Vertex AI"""

    def __init__(self, model_name: str, project_id: str, model_type: str = None):
        self.model_name = model_name
        self.project_id = project_id
        self.model_type = model_type

    def prepare_request(self, **kwargs) -> Dict[str, Any]:
        # Adapta o modelo baseado no model_type
        if self.model_type:
            if self.model_type == "GoogleVertexClaude":
                kwargs['model'] = f"vertex_ai/claude-{self.model_name}"
            elif self.model_type == "GoogleVertexDeepSeek":
                kwargs['model'] = f"vertex_ai/deepseek-{self.model_name}"
            elif self.model_type == "GoogleVertexMeta":
                kwargs['model'] = f"vertex_ai/llama-{self.model_name}"
            elif self.model_type == "GoogleVertexMistral":
                kwargs['model'] = f"vertex_ai/mistral-{self.model_name}"
            else:
                kwargs['model'] = f"vertex_ai/{self.model_name}"
        else:
            kwargs['model'] = f"vertex_ai/{self.model_name}"

        kwargs['vertex_project'] = self.project_id
        return kwargs

    def parse_response(self, response: Any) -> Dict[str, Any]:
        if hasattr(response, 'model_dump'):
            return response.model_dump()
        return dict(response)
```


infra/litellm/utils.py
```python
import hashlib
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

class TokenCounter:
    """Utilitário para contagem de tokens"""

    @staticmethod
    def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
        """Estima a contagem de tokens para um texto"""
        # Estimativa simples: ~4 caracteres por token
        return len(text) // 4

    @staticmethod
    def count_messages_tokens(messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo") -> int:
        """Conta tokens em uma lista de mensagens"""
        total = 0
        for message in messages:
            total += TokenCounter.count_tokens(message.get('content', ''), model)
            total += 4  # Overhead por mensagem
        return total

class CacheManager:
    """Gerenciador de cache para respostas LLM"""

    def __init__(self, ttl_minutes: int = 15):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = timedelta(minutes=ttl_minutes)

    def _generate_key(self, **kwargs) -> str:
        """Gera chave de cache baseada nos parâmetros"""
        # Remove parâmetros não determinísticos
        cache_params = {k: v for k, v in kwargs.items()
                       if k not in ['stream', 'user', 'request_id']}
        key_str = json.dumps(cache_params, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Busca resposta no cache"""
        key = self._generate_key(**kwargs)
        if key in self.cache:
            entry = self.cache[key]
            if datetime.utcnow() - entry['timestamp'] < self.ttl:
                return entry['response']
            else:
                del self.cache[key]
        return None

    def set(self, response: Dict[str, Any], **kwargs):
        """Armazena resposta no cache"""
        key = self._generate_key(**kwargs)
        self.cache[key] = {
            'response': response,
            'timestamp': datetime.utcnow()
        }

    def clear_expired(self):
        """Remove entradas expiradas do cache"""
        now = datetime.utcnow()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now - entry['timestamp'] >= self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]

class RateLimiter:
    """Limitador de taxa para requisições"""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = timedelta(seconds=window_seconds)
        self.requests: Dict[str, List[datetime]] = {}

    def is_allowed(self, client_id: str) -> bool:
        """Verifica se o cliente pode fazer uma requisição"""
        now = datetime.utcnow()

        if client_id not in self.requests:
            self.requests[client_id] = []

        # Remove requisições antigas
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < self.window
        ]

        # Verifica limite
        if len(self.requests[client_id]) < self.max_requests:
            self.requests[client_id].append(now)
            return True

        return False

    def get_reset_time(self, client_id: str) -> Optional[datetime]:
        """Retorna quando o limite será resetado"""
        if client_id in self.requests and self.requests[client_id]:
            oldest_request = min(self.requests[client_id])
            return oldest_request + self.window
        return None

class OperationValidator:
    """Validador de operações suportadas"""

    VALID_OPERATIONS = [
        "Responses",
        "ChatCompletion",
        "embeddings",
        "images",
        "audio_tts",
        "audio_stt",
        "audio_realtime"
    ]

    @classmethod
    def validate(cls, operations: List[str]) -> bool:
        """Valida se todas as operações são válidas"""
        return all(op in cls.VALID_OPERATIONS for op in operations)

    @classmethod
    def get_endpoint_for_operation(cls, operation: str) -> str:
        """Retorna o endpoint correspondente para uma operação"""
        mapping = {
            "Responses": "/v1/chat/completions",
            "ChatCompletion": "/v1/chat/completions",
            "embeddings": "/v1/embeddings",
            "images": "/v1/images/generations",
            "audio_tts": "/v1/audio/speech",
            "audio_stt": "/v1/audio/transcriptions",
            "audio_realtime": "/v1/audio/realtime"
        }
        return mapping.get(operation, "")
```


requirements.txt
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
azure-cosmos==4.5.1
azure-identity==1.15.0
litellm==1.17.0
python-multipart==0.0.6
httpx==0.25.2
python-dotenv==1.0.0
```


Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "infra.controllers.main:app", "--host", "0.0.0.0", "--port", "8000"]
```


docker-compose.yml
```yaml
version: '3.8'

services:
  llm-proxy:
    build: .
    ports:
      - "8000:8000"
    environment:
      - COSMOS_URL=${COSMOS_URL}
      - COSMOS_KEY=${COSMOS_KEY}
      - COSMOS_DATABASE_ID=${COSMOS_DATABASE_ID:-llm-proxy}
      - COSMOS_CONTAINER_ID=${COSMOS_CONTAINER_ID:-models}
      - ENVIRONMENT=${ENVIRONMENT:-development}
      # Provider API Keys (examples)
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - .:/app
    working_dir: /app
    command: uvicorn infra.controllers.main:app --host 0.0.0.0 --port 8000 --reload
    restart: unless-stopped

  # Optional: Add a reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - llm-proxy
    restart: unless-stopped
```


.env.example
# Cosmos DB Configuration
COSMOS_URL=https://your-cosmosdb-account.documents.azure.com:443/
COSMOS_KEY=your-cosmosdb-primary-key
COSMOS_DATABASE_ID=llm-proxy
COSMOS_CONTAINER_ID=models

# Environment
ENVIRONMENT=development
HOST=0.0.0.0
PORT=8000

# Provider API Keys (examples based on your model parameters)
OPENAI_API_KEY=sk-your-openai-key
AZURE_OPENAI_API_KEY=your-azure-openai-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
ANTHROPIC_API_KEY=your-anthropic-key

# Custom provider configurations
CUSTOM_API_KEY=your-custom-provider-key
CUSTOM_API_ENDPOINT=https://api.customprovider.com

## 📊 Estruturas de Dados

### Modelo no Cosmos DB

Cada modelo cadastrado no Cosmos DB deve seguir esta estrutura:
```json
// Exemplo 1: GPT-4 via Azure OpenAI
{
  "id": "gpt-4-azure",
  "projects": ["project1", "project2"],
  "private": false,
  "provider": "Azure",
  "model_type": "AzureOpenAI",
  "name": "GPT-4 Azure OpenAI",
  "parameters": {
    "secret_name": "AZURE_OPENAI_SECRET",
    "deployment_name": "gpt-4-deployment",
    "api_version": "2024-02-01",
    "endpoint": "https://meu-recurso.openai.azure.com/",
    "enable_tools": true
  },
  "costs": {
    "input": 0.03,   // $0.03 por 1k tokens de entrada
    "output": 0.06   // $0.06 por 1k tokens de saída
  },
  "operations": ["ChatCompletion", "Responses", "embeddings"]
}

// Exemplo 2: DeepSeek via Google Vertex AI
{
  "id": "deepseek-vertex",
  "projects": [],
  "private": false,
  "provider": "Google",
  "model_type": "GoogleVertexDeepSeek",
  "name": "DeepSeek V2 via Vertex AI",
  "parameters": {
    "secret_name": "SP-BRIDGE-4852-DEV",
    "gcp_project": "meu-projeto-gcp",
    "location": "us-central1",
    "enable_tools": true
  },
  "costs": {
    "input": 0.014,  // $0.014 por 1k tokens de entrada
    "output": 0.028  // $0.028 por 1k tokens de saída
  },
  "operations": ["ChatCompletion", "Responses"]
}

// Exemplo 3: Claude via Google Vertex AI
{
  "id": "claude-vertex",
  "projects": [],
  "private": false,
  "provider": "Google",
  "model_type": "GoogleVertexClaude",
  "name": "Claude 3 Sonnet via Vertex AI",
  "parameters": {
    "secret_name": "SP-BRIDGE-4852-DEV",
    "gcp_project": "meu-projeto-gcp",
    "location": "us-central1",
    "enable_tools": true
  },
  "costs": {
    "input": 0.003,  // $0.003 por 1k tokens de entrada
    "output": 0.015  // $0.015 por 1k tokens de saída
  },
  "operations": ["ChatCompletion", "Responses", "images"]
}

// Exemplo 4: Modelo com todas as operações
{
  "id": "gpt-4-omni",
  "projects": [],
  "private": false,
  "provider": "Azure",
  "model_type": "AzureOpenAI",
  "name": "GPT-4 Omni - Completo",
  "parameters": {
    "secret_name": "AZURE_OPENAI_SECRET",
    "deployment_name": "gpt-4-omni",
    "api_version": "2024-02-01",
    "endpoint": "https://meu-recurso.openai.azure.com/",
    "enable_tools": true
  },
  "costs": {
    "input": 0.005,
    "output": 0.015
  },
  "operations": [
    "Responses", 
    "ChatCompletion", 
    "embeddings", 
    "images", 
    "audio_tts", 
    "audio_stt", 
    "audio_realtime"
  ]
}
```

### Tipos de Modelos Suportados (model_type)

| model_type | Descrição | Provider | Exemplo de Modelo |
|------------|-----------|----------|-------------------|
| `AzureOpenAI` | Azure OpenAI Service | azure | GPT-4, GPT-3.5-turbo |
| `AzureFoundry` | Azure AI Foundry | azure | Modelos via Foundry |
| `GoogleVertex` | Google Vertex AI nativo | google | Gemini Pro, PaLM |
| `GoogleVertexClaude` | Claude via Vertex AI | google | Claude 3 Opus/Sonnet |
| `GoogleVertexDeepSeek` | DeepSeek via Vertex AI | google | DeepSeek V2, DeepSeek Coder |
| `GoogleVertexMeta` | Meta LLaMA via Vertex AI | google | LLaMA 2, Code Llama |
| `GoogleVertexMistral` | Mistral via Vertex AI | google | Mistral 7B, Mixtral |
| `GoogleVertexJamba` | AI21 Jamba via Vertex AI | google | Jamba 1.5 |
| `GoogleVertexQwen` | Alibaba Qwen via Vertex AI | google | Qwen 2, Qwen-Coder |
| `GoogleVertexOpenAI` | OpenAI via Vertex AI | google | GPT-4, GPT-3.5 |

### Endpoints por Funcionalidade

| Funcionalidade | Endpoint | model_type Suportados |
|---------------|----------|----------------------|
| Chat/Completions | `/v1/chat/completions` | Todos os tipos acima |
| Image Generation | `/v1/images/generations` | AzureOpenAI, GoogleVertex |
| Text-to-Speech | `/v1/audio/speech` | AzureOpenAI, GoogleVertex |
| Speech-to-Text | `/v1/audio/transcriptions` | AzureOpenAI, GoogleVertex |

## 🔌 API Reference

### Endpoints Disponíveis

#### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "environment": "development"
}
```

#### 2. Listar Modelos
```http
GET /v1/models
```
**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-4",
      "object": "model",
      "owned_by": "azure"
    }
  ]
}
```

#### 3. Chat Completions
```http
POST /v1/chat/completions
Content-Type: application/json
```
**Request:**
```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 150,
  "stream": false
}
```

#### 4. Text-to-Speech
```http
POST /v1/audio/speech
Content-Type: application/json
```
**Request:**
```json
{
  "model": "tts-1",
  "input": "Hello world!",
  "voice": "alloy",
  "response_format": "mp3"
}
```

#### 5. Speech-to-Text
```http
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
```
**Request:**
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "language=pt"
```

#### 6. Image Generation
```http
POST /v1/images/generations
Content-Type: application/json
```
**Request:**
```json
{
  "model": "dall-e-3",
  "prompt": "A beautiful sunset over mountains",
  "n": 1,
  "size": "1024x1024"
}
```

## 🧪 Testes

### Testes Unitários

```bash
# Instalar dependências de teste
pip install pytest pytest-asyncio pytest-cov httpx

# Executar todos os testes
pytest

# Com cobertura
pytest --cov=. --cov-report=html

# Testes específicos
pytest tests/test_chat_completions.py -v
```

### Teste de Carga

```bash
# Usando Apache Bench
ab -n 1000 -c 10 -p request.json -T application/json \
   http://localhost:8000/v1/chat/completions

# Usando Locust
locust -f tests/locustfile.py --host=http://localhost:8000
```

## 🔧 Troubleshooting

### Problemas Comuns e Soluções

#### 1. Erro de Conexão com Cosmos DB
```
Error: Unable to connect to Cosmos DB
```
**Solução:**
- Verifique as credenciais no .env
- Confirme que o firewall do Cosmos DB permite seu IP
- Teste a conexão: `az cosmosdb show --name sua-conta`

#### 2. Modelo não encontrado
```
Error: Model 'gpt-4' not found
```
**Solução:**
- Verifique se o modelo está cadastrado no Cosmos DB
- Use `GET /v1/models` para listar modelos disponíveis
- Confirme que o campo `code` corresponde ao solicitado

#### 3. API Key inválida
```
Error: Invalid API key for provider
```
**Solução:**
- Verifique a variável de ambiente correspondente
- Teste a API key diretamente: `curl -H "Authorization: Bearer $OPENAI_API_KEY" ...`
- Confirme que a API key tem as permissões necessárias

#### 4. Timeout em requisições
```
Error: Request timeout
```
**Solução:**
- Aumente o timeout no LiteLLM: adicione `"timeout": "300"` nos parameters
- Considere usar streaming para respostas longas
- Verifique a latência de rede com o provider

## 🚢 Deploy em Produção

### Deploy no Azure Container Instances

```bash
# Build da imagem
docker build -t llm-proxy:latest .

# Tag para Azure Container Registry
docker tag llm-proxy:latest seu-registry.azurecr.io/llm-proxy:latest

# Push para ACR
az acr login --name seu-registry
docker push seu-registry.azurecr.io/llm-proxy:latest

# Deploy no ACI
az container create \
  --resource-group seu-rg \
  --name llm-proxy \
  --image seu-registry.azurecr.io/llm-proxy:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables \
    COSMOS_URL=$COSMOS_URL \
    COSMOS_KEY=$COSMOS_KEY \
  --secure-environment-variables \
    OPENAI_API_KEY=$OPENAI_API_KEY
```

### Deploy no Kubernetes

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-proxy
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-proxy
  template:
    metadata:
      labels:
        app: llm-proxy
    spec:
      containers:
      - name: llm-proxy
        image: seu-registry.azurecr.io/llm-proxy:latest
        ports:
        - containerPort: 8000
        env:
        - name: COSMOS_URL
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: cosmos-url
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

## 📈 Monitoramento

### Integração com Application Insights

```python
# Adicione ao requirements.txt
opencensus-ext-azure==1.1.9
opencensus-ext-logging==0.1.1

# Configure no main.py
from opencensus.ext.azure import metrics_exporter
from opencensus.stats import aggregation, measure, stats, view

exporter = metrics_exporter.new_metrics_exporter(
    connection_string='InstrumentationKey=sua-key'
)
```

### Métricas Importantes

- **Latência de Requisições**: P50, P95, P99
- **Taxa de Erro**: Erros por minuto
- **Uso de Tokens**: Tokens consumidos por provider
- **Cache Hit Rate**: Taxa de acerto do cache (se implementado)
- **Disponibilidade**: Uptime do serviço

## 🔒 Segurança

### Boas Práticas Implementadas

1. **Gestão de Secrets**
   - Nunca commitar API keys
   - Usar Azure Key Vault em produção
   - Rotação regular de credenciais

2. **Validação de Entrada**
   - DTOs com Pydantic para validação
   - Limites de tamanho em requisições
   - Sanitização de inputs

3. **Rate Limiting** (Adicionar com FastAPI-Limiter)
   ```python
   from fastapi_limiter import FastAPILimiter
   from fastapi_limiter.depends import RateLimiter
   
   @app.post("/v1/chat/completions", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
   ```

4. **CORS Configuration**
   ```python
   from fastapi.middleware.cors import CORSMiddleware
   
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://seu-dominio.com"],
       allow_methods=["GET", "POST"],
       allow_headers=["*"],
   )
   ```

## 📦 Estrutura de Dados do Modelo

### Provider como String

No sistema LLM Proxy, o **provider é simplesmente um campo string** dentro do documento do modelo no Cosmos DB. Não há uma entidade separada para providers.

**Exemplo de documento no Cosmos DB:**
```json
{
  "id": "gpt-4-azure",
  "name": "GPT-4 Azure",
  "code": "gpt-4",
  "provider": "Azure",  // <-- Provider é apenas uma string
  "model_type": "AzureOpenAI",
  "parameters": {
    "api_key": "AZURE_OPENAI_API_KEY",
    "deployment_name": "gpt-4-deployment"
  }
}
```

### Providers e Model Types Suportados

| Provider | Model Types Disponíveis | Descrição |
|----------|------------------------|-------------|
| `"Azure"` | `AzureOpenAI`, `AzureFoundry` | Azure OpenAI Service e AI Foundry |
| `"Google"` | `GoogleVertex`, `GoogleVertexClaude`, `GoogleVertexDeepSeek`, `GoogleVertexMeta`, `GoogleVertexMistral`, `GoogleVertexJamba`, `GoogleVertexQwen`, `GoogleVertexOpenAI` | Google Vertex AI com diferentes modelos |

### Operations Suportadas

| Operation | Descrição | Endpoint Correspondente |
|-----------|-----------|------------------------|
| `Responses` | Respostas básicas de texto | `/v1/chat/completions` |
| `ChatCompletion` | Chat completions completas | `/v1/chat/completions` |
| `embeddings` | Geração de embeddings | `/v1/embeddings` |
| `images` | Geração/análise de imagens | `/v1/images/generations` |
| `audio_tts` | Text-to-Speech | `/v1/audio/speech` |
| `audio_stt` | Speech-to-Text | `/v1/audio/transcriptions` |
| `audio_realtime` | Áudio em tempo real | `/v1/audio/realtime` |

## 🎯 Características Principais

| Feature | Descrição | Status |
|---------|-----------|--------|
| **Arquitetura Hexagonal** | Separação clara entre domínio, aplicação e infraestrutura | ✅ Implementado |
| **OpenAI Compatibility** | 100% compatível com a API OpenAI | ✅ Implementado |
| **Multi-Provider** | Azure, Google, Anthropic, OpenAI via LiteLLM | ✅ Implementado |
| **Configuration Management** | Configuração via variáveis de ambiente | ✅ Implementado |
| **Model Management** | Gestão centralizada no Cosmos DB | ✅ Implementado |
| **Extensibilidade** | Fácil adição de novos provedores | ✅ Implementado |
| **Error Handling** | Tratamento robusto de erros | ✅ Implementado |
| **Health Checks** | Monitoramento de saúde | ✅ Implementado |
| **Rate Limiting** | Controle de taxa de requisições | 🔄 Planejado |
| **Caching** | Cache de respostas | 🔄 Planejado |
| **Metrics & Logging** | Observabilidade completa | 🔄 Em desenvolvimento |

## 📚 Recursos Adicionais

### Documentação Relacionada
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Azure Cosmos DB Documentation](https://docs.microsoft.com/azure/cosmos-db/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

### Exemplos de Integração

#### Python Client
```python
import openai

# Configure para usar seu proxy
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "dummy-key"  # O proxy gerencia as keys reais

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

#### JavaScript/Node.js
```javascript
const OpenAI = require('openai');

const openai = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'dummy-key',
});

const response = await openai.chat.completions.create({
  model: 'gpt-4',
  messages: [{ role: 'user', content: 'Hello!' }],
});
```

## 🤝 Contribuindo

1. Fork o projeto
2. Crie sua feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está licenciado sob a MIT License.

## 👥 Suporte

Para suporte, abra uma issue no GitHub ou entre em contato através do email: suporte@llm-proxy.com

---

**Desenvolvido com ❤️ usando FastAPI e Python**
