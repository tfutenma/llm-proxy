# Arquitetura Hexagonal - Exemplo Prático Completo

## 🎯 Domínio de Exemplo: Sistema de Biblioteca Digital

Vamos implementar um sistema de biblioteca digital para demonstrar todos os conceitos da arquitetura hexagonal. Este exemplo é suficientemente complexo para mostrar todos os padrões, mas simples o suficiente para ser compreendido facilmente.

### 📋 Requisitos do Sistema

**Funcionalidades Core:**
- Gerenciar livros (criar, buscar, atualizar)
- Gerenciar empréstimos de livros
- Notificar usuários sobre vencimentos
- Gerar relatórios de empréstimos

**Regras de Negócio:**
- Um livro pode ter múltiplas cópias
- Empréstimos têm prazo de 14 dias
- Usuários podem ter no máximo 3 livros emprestados
- Notificações são enviadas 2 dias antes do vencimento

## 🏗️ Racional da Arquitetura Hexagonal

### Por que Arquitetura Hexagonal?

1. **Independência de Framework**: O domínio não depende de Flask, FastAPI, Django, etc.
2. **Testabilidade**: Podemos testar a lógica de negócio sem banco de dados ou APIs
3. **Flexibilidade**: Podemos trocar banco de dados, APIs, ou interfaces sem afetar o core
4. **Manutenibilidade**: Código organizado em camadas com responsabilidades claras

### Estrutura das Camadas

```
📁 library_system/
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
│   ├── persistence/        # Adaptadores de banco de dados
│   ├── web/                # Adaptadores HTTP
│   ├── notifications/      # Adaptadores de notificação
│   └── config/             # Configurações
└── 📁 tests/               # Testes automatizados
    ├── unit/               # Testes unitários
    ├── integration/        # Testes de integração
    └── e2e/                # Testes end-to-end
```

## 🔧 Implementação Completa

### 1. Camada de Domínio (Core)

#### `domain/entities/book.py`
```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from ..value_objects.isbn import ISBN
from ..value_objects.book_status import BookStatus

@dataclass
class Book:
    """Entidade central do domínio: Livro"""

    id: UUID = field(default_factory=uuid4)
    title: str = ""
    author: str = ""
    isbn: ISBN = field(default_factory=lambda: ISBN(""))
    total_copies: int = 0
    available_copies: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.title:
            raise ValueError("Título é obrigatório")
        if not self.author:
            raise ValueError("Autor é obrigatório")
        if self.total_copies < 0:
            raise ValueError("Total de cópias deve ser positivo")
        if self.available_copies > self.total_copies:
            raise ValueError("Cópias disponíveis não pode exceder total")

    def is_available(self) -> bool:
        """Verifica se há cópias disponíveis para empréstimo"""
        return self.available_copies > 0

    def borrow_copy(self) -> None:
        """Empresta uma cópia do livro"""
        if not self.is_available():
            raise ValueError(f"Livro '{self.title}' não possui cópias disponíveis")
        self.available_copies -= 1
        self.updated_at = datetime.utcnow()

    def return_copy(self) -> None:
        """Devolve uma cópia do livro"""
        if self.available_copies >= self.total_copies:
            raise ValueError(f"Todas as cópias do livro '{self.title}' já foram devolvidas")
        self.available_copies += 1
        self.updated_at = datetime.utcnow()

    def add_copies(self, quantity: int) -> None:
        """Adiciona cópias ao acervo"""
        if quantity <= 0:
            raise ValueError("Quantidade deve ser positiva")
        self.total_copies += quantity
        self.available_copies += quantity
        self.updated_at = datetime.utcnow()

    def get_status(self) -> BookStatus:
        """Retorna o status atual do livro"""
        if self.available_copies == 0:
            return BookStatus.UNAVAILABLE
        elif self.available_copies == self.total_copies:
            return BookStatus.AVAILABLE
        else:
            return BookStatus.PARTIALLY_AVAILABLE
```

#### `domain/entities/user.py`
```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List
from uuid import UUID, uuid4

from ..value_objects.email import Email

@dataclass
class User:
    """Entidade: Usuário da biblioteca"""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    email: Email = field(default_factory=lambda: Email(""))
    max_borrowed_books: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.name:
            raise ValueError("Nome é obrigatório")
        if self.max_borrowed_books <= 0:
            raise ValueError("Limite de livros deve ser positivo")

    def can_borrow_more_books(self, current_loans: int) -> bool:
        """Verifica se o usuário pode emprestar mais livros"""
        return current_loans < self.max_borrowed_books

    def get_remaining_loan_capacity(self, current_loans: int) -> int:
        """Retorna quantos livros o usuário ainda pode emprestar"""
        return max(0, self.max_borrowed_books - current_loans)
```

#### `domain/entities/loan.py`
```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID, uuid4

from ..value_objects.loan_status import LoanStatus

@dataclass
class Loan:
    """Entidade: Empréstimo de livro"""

    id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    book_id: UUID = field(default_factory=uuid4)
    loan_date: datetime = field(default_factory=datetime.utcnow)
    due_date: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=14))
    return_date: Optional[datetime] = None
    renewed_count: int = 0
    max_renewals: int = 2

    def __post_init__(self):
        if self.due_date <= self.loan_date:
            raise ValueError("Data de vencimento deve ser posterior à data do empréstimo")

    def is_active(self) -> bool:
        """Verifica se o empréstimo está ativo"""
        return self.return_date is None

    def is_overdue(self) -> bool:
        """Verifica se o empréstimo está vencido"""
        return self.is_active() and datetime.utcnow() > self.due_date

    def days_until_due(self) -> int:
        """Retorna quantos dias faltam para o vencimento"""
        if not self.is_active():
            return 0
        delta = self.due_date - datetime.utcnow()
        return max(0, delta.days)

    def can_renew(self) -> bool:
        """Verifica se o empréstimo pode ser renovado"""
        return (self.is_active() and
                self.renewed_count < self.max_renewals and
                not self.is_overdue())

    def renew(self, additional_days: int = 14) -> None:
        """Renova o empréstimo"""
        if not self.can_renew():
            raise ValueError("Empréstimo não pode ser renovado")

        self.due_date += timedelta(days=additional_days)
        self.renewed_count += 1

    def return_book(self) -> None:
        """Registra a devolução do livro"""
        if not self.is_active():
            raise ValueError("Empréstimo já foi finalizado")

        self.return_date = datetime.utcnow()

    def get_status(self) -> LoanStatus:
        """Retorna o status atual do empréstimo"""
        if not self.is_active():
            return LoanStatus.RETURNED
        elif self.is_overdue():
            return LoanStatus.OVERDUE
        elif self.days_until_due() <= 2:
            return LoanStatus.DUE_SOON
        else:
            return LoanStatus.ACTIVE
```

#### `domain/value_objects/isbn.py`
```python
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class ISBN:
    """Value Object: ISBN do livro"""

    value: str

    def __post_init__(self):
        if not self.value:
            raise ValueError("ISBN é obrigatório")

        # Remove hífens e espaços
        clean_isbn = re.sub(r'[-\s]', '', self.value)

        if not self._is_valid_isbn(clean_isbn):
            raise ValueError(f"ISBN inválido: {self.value}")

        # Armazena versão limpa
        object.__setattr__(self, 'value', clean_isbn)

    def _is_valid_isbn(self, isbn: str) -> bool:
        """Valida formato do ISBN"""
        # ISBN-10 ou ISBN-13
        if len(isbn) == 10:
            return self._is_valid_isbn10(isbn)
        elif len(isbn) == 13:
            return self._is_valid_isbn13(isbn)
        return False

    def _is_valid_isbn10(self, isbn: str) -> bool:
        """Valida ISBN-10"""
        if not re.match(r'^\d{9}[\dX]$', isbn):
            return False

        total = 0
        for i, char in enumerate(isbn[:-1]):
            total += int(char) * (10 - i)

        check_digit = isbn[-1]
        if check_digit == 'X':
            check_digit = 10
        else:
            check_digit = int(check_digit)

        return (total + check_digit) % 11 == 0

    def _is_valid_isbn13(self, isbn: str) -> bool:
        """Valida ISBN-13"""
        if not re.match(r'^\d{13}$', isbn):
            return False

        total = 0
        for i, char in enumerate(isbn[:-1]):
            multiplier = 1 if i % 2 == 0 else 3
            total += int(char) * multiplier

        check_digit = int(isbn[-1])
        return (total + check_digit) % 10 == 0

    def formatted(self) -> str:
        """Retorna ISBN formatado"""
        if len(self.value) == 10:
            return f"{self.value[:1]}-{self.value[1:6]}-{self.value[6:9]}-{self.value[9:]}"
        else:
            return f"{self.value[:3]}-{self.value[3:4]}-{self.value[4:6]}-{self.value[6:12]}-{self.value[12:]}"
```

#### `domain/value_objects/email.py`
```python
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class Email:
    """Value Object: Email do usuário"""

    value: str

    def __post_init__(self):
        if not self.value:
            raise ValueError("Email é obrigatório")

        if not self._is_valid_email(self.value):
            raise ValueError(f"Email inválido: {self.value}")

    def _is_valid_email(self, email: str) -> bool:
        """Valida formato do email"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def domain(self) -> str:
        """Retorna o domínio do email"""
        return self.value.split('@')[1]

    def local_part(self) -> str:
        """Retorna a parte local do email"""
        return self.value.split('@')[0]
```

#### `domain/value_objects/enums.py`
```python
from enum import Enum

class BookStatus(Enum):
    """Status de disponibilidade do livro"""
    AVAILABLE = "available"
    PARTIALLY_AVAILABLE = "partially_available"
    UNAVAILABLE = "unavailable"

class LoanStatus(Enum):
    """Status do empréstimo"""
    ACTIVE = "active"
    DUE_SOON = "due_soon"
    OVERDUE = "overdue"
    RETURNED = "returned"

class NotificationType(Enum):
    """Tipos de notificação"""
    DUE_REMINDER = "due_reminder"
    OVERDUE_NOTICE = "overdue_notice"
    RETURN_CONFIRMATION = "return_confirmation"
    RENEWAL_CONFIRMATION = "renewal_confirmation"
```

#### `domain/services/loan_service.py`
```python
from datetime import datetime, timedelta
from typing import List
from uuid import UUID

from ..entities.loan import Loan
from ..entities.user import User
from ..entities.book import Book
from ..repositories.loan_repository import LoanRepositoryPort

class LoanDomainService:
    """Serviço de domínio para regras complexas de empréstimo"""

    def __init__(self, loan_repository: LoanRepositoryPort):
        self._loan_repository = loan_repository

    def can_user_borrow_book(self, user: User, book: Book) -> tuple[bool, str]:
        """Verifica se usuário pode emprestar o livro"""
        # Verifica se o livro está disponível
        if not book.is_available():
            return False, f"Livro '{book.title}' não possui cópias disponíveis"

        # Conta empréstimos ativos do usuário
        active_loans = self._loan_repository.count_active_loans_by_user(user.id)

        # Verifica limite de empréstimos
        if not user.can_borrow_more_books(active_loans):
            return False, f"Usuário já possui {active_loans} livros emprestados (limite: {user.max_borrowed_books})"

        # Verifica se usuário já tem este livro emprestado
        has_same_book = self._loan_repository.user_has_active_loan_for_book(user.id, book.id)
        if has_same_book:
            return False, f"Usuário já possui o livro '{book.title}' emprestado"

        return True, "Empréstimo permitido"

    def create_loan(self, user: User, book: Book) -> Loan:
        """Cria um novo empréstimo"""
        can_borrow, message = self.can_user_borrow_book(user, book)

        if not can_borrow:
            raise ValueError(message)

        # Cria o empréstimo
        loan = Loan(
            user_id=user.id,
            book_id=book.id
        )

        # Atualiza disponibilidade do livro
        book.borrow_copy()

        return loan

    def return_loan(self, loan: Loan, book: Book) -> None:
        """Processa devolução de empréstimo"""
        if not loan.is_active():
            raise ValueError("Empréstimo já foi finalizado")

        loan.return_book()
        book.return_copy()

    def get_loans_due_soon(self, days_ahead: int = 2) -> List[Loan]:
        """Retorna empréstimos que vencem em X dias"""
        return self._loan_repository.find_loans_due_in_days(days_ahead)

    def get_overdue_loans(self) -> List[Loan]:
        """Retorna empréstimos vencidos"""
        return self._loan_repository.find_overdue_loans()
```

### 2. Camada de Aplicação (Use Cases)

#### `application/ports/repositories.py`
```python
from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from domain.entities.book import Book
from domain.entities.user import User
from domain.entities.loan import Loan

class BookRepositoryPort(ABC):
    """Port para repositório de livros"""

    @abstractmethod
    async def save(self, book: Book) -> Book:
        pass

    @abstractmethod
    async def find_by_id(self, book_id: UUID) -> Optional[Book]:
        pass

    @abstractmethod
    async def find_by_isbn(self, isbn: str) -> Optional[Book]:
        pass

    @abstractmethod
    async def find_by_title(self, title: str) -> List[Book]:
        pass

    @abstractmethod
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[Book]:
        pass

    @abstractmethod
    async def delete(self, book_id: UUID) -> bool:
        pass

class UserRepositoryPort(ABC):
    """Port para repositório de usuários"""

    @abstractmethod
    async def save(self, user: User) -> User:
        pass

    @abstractmethod
    async def find_by_id(self, user_id: UUID) -> Optional[User]:
        pass

    @abstractmethod
    async def find_by_email(self, email: str) -> Optional[User]:
        pass

    @abstractmethod
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[User]:
        pass

class LoanRepositoryPort(ABC):
    """Port para repositório de empréstimos"""

    @abstractmethod
    async def save(self, loan: Loan) -> Loan:
        pass

    @abstractmethod
    async def find_by_id(self, loan_id: UUID) -> Optional[Loan]:
        pass

    @abstractmethod
    async def find_by_user_id(self, user_id: UUID) -> List[Loan]:
        pass

    @abstractmethod
    async def find_active_by_user_id(self, user_id: UUID) -> List[Loan]:
        pass

    @abstractmethod
    async def count_active_loans_by_user(self, user_id: UUID) -> int:
        pass

    @abstractmethod
    async def user_has_active_loan_for_book(self, user_id: UUID, book_id: UUID) -> bool:
        pass

    @abstractmethod
    async def find_loans_due_in_days(self, days: int) -> List[Loan]:
        pass

    @abstractmethod
    async def find_overdue_loans(self) -> List[Loan]:
        pass
```

#### `application/ports/notifications.py`
```python
from abc import ABC, abstractmethod
from typing import Dict, Any

from domain.value_objects.enums import NotificationType

class NotificationPort(ABC):
    """Port para serviço de notificações"""

    @abstractmethod
    async def send_notification(
        self,
        recipient_email: str,
        notification_type: NotificationType,
        context: Dict[str, Any]
    ) -> bool:
        pass

class EmailServicePort(ABC):
    """Port para serviço de email"""

    @abstractmethod
    async def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        html_body: str = None
    ) -> bool:
        pass
```

#### `application/use_cases/book_management.py`
```python
from typing import List, Optional
from uuid import UUID

from domain.entities.book import Book
from domain.value_objects.isbn import ISBN
from ..ports.repositories import BookRepositoryPort

class CreateBookUseCase:
    """Caso de uso: Criar livro"""

    def __init__(self, book_repository: BookRepositoryPort):
        self._book_repository = book_repository

    async def execute(
        self,
        title: str,
        author: str,
        isbn: str,
        total_copies: int
    ) -> Book:
        """Executa a criação de um livro"""

        # Verifica se já existe livro com o mesmo ISBN
        existing_book = await self._book_repository.find_by_isbn(isbn)
        if existing_book:
            raise ValueError(f"Já existe um livro com ISBN {isbn}")

        # Cria o livro
        book = Book(
            title=title,
            author=author,
            isbn=ISBN(isbn),
            total_copies=total_copies,
            available_copies=total_copies
        )

        # Salva no repositório
        return await self._book_repository.save(book)

class FindBookUseCase:
    """Caso de uso: Buscar livros"""

    def __init__(self, book_repository: BookRepositoryPort):
        self._book_repository = book_repository

    async def execute_by_id(self, book_id: UUID) -> Optional[Book]:
        """Busca livro por ID"""
        return await self._book_repository.find_by_id(book_id)

    async def execute_by_title(self, title: str) -> List[Book]:
        """Busca livros por título"""
        return await self._book_repository.find_by_title(title)

    async def execute_by_isbn(self, isbn: str) -> Optional[Book]:
        """Busca livro por ISBN"""
        return await self._book_repository.find_by_isbn(isbn)

class UpdateBookCopiesUseCase:
    """Caso de uso: Atualizar cópias do livro"""

    def __init__(self, book_repository: BookRepositoryPort):
        self._book_repository = book_repository

    async def execute(self, book_id: UUID, additional_copies: int) -> Book:
        """Adiciona cópias ao acervo"""

        book = await self._book_repository.find_by_id(book_id)
        if not book:
            raise ValueError(f"Livro com ID {book_id} não encontrado")

        book.add_copies(additional_copies)

        return await self._book_repository.save(book)
```

#### `application/use_cases/loan_management.py`
```python
from typing import List
from uuid import UUID

from domain.entities.loan import Loan
from domain.services.loan_service import LoanDomainService
from ..ports.repositories import BookRepositoryPort, UserRepositoryPort, LoanRepositoryPort
from ..ports.notifications import NotificationPort
from domain.value_objects.enums import NotificationType

class BorrowBookUseCase:
    """Caso de uso: Emprestar livro"""

    def __init__(
        self,
        book_repository: BookRepositoryPort,
        user_repository: UserRepositoryPort,
        loan_repository: LoanRepositoryPort,
        notification_service: NotificationPort
    ):
        self._book_repository = book_repository
        self._user_repository = user_repository
        self._loan_repository = loan_repository
        self._notification_service = notification_service
        self._loan_service = LoanDomainService(loan_repository)

    async def execute(self, user_id: UUID, book_id: UUID) -> Loan:
        """Executa o empréstimo de um livro"""

        # Busca usuário e livro
        user = await self._user_repository.find_by_id(user_id)
        if not user:
            raise ValueError(f"Usuário com ID {user_id} não encontrado")

        book = await self._book_repository.find_by_id(book_id)
        if not book:
            raise ValueError(f"Livro com ID {book_id} não encontrado")

        # Cria o empréstimo usando o serviço de domínio
        loan = self._loan_service.create_loan(user, book)

        # Salva o empréstimo e atualiza o livro
        saved_loan = await self._loan_repository.save(loan)
        await self._book_repository.save(book)

        # Envia notificação
        await self._notification_service.send_notification(
            recipient_email=user.email.value,
            notification_type=NotificationType.RETURN_CONFIRMATION,
            context={
                "user_name": user.name,
                "book_title": book.title,
                "due_date": loan.due_date.strftime("%d/%m/%Y")
            }
        )

        return saved_loan

class ReturnBookUseCase:
    """Caso de uso: Devolver livro"""

    def __init__(
        self,
        book_repository: BookRepositoryPort,
        loan_repository: LoanRepositoryPort,
        notification_service: NotificationPort
    ):
        self._book_repository = book_repository
        self._loan_repository = loan_repository
        self._notification_service = notification_service
        self._loan_service = LoanDomainService(loan_repository)

    async def execute(self, loan_id: UUID) -> Loan:
        """Executa a devolução de um livro"""

        # Busca o empréstimo
        loan = await self._loan_repository.find_by_id(loan_id)
        if not loan:
            raise ValueError(f"Empréstimo com ID {loan_id} não encontrado")

        # Busca o livro
        book = await self._book_repository.find_by_id(loan.book_id)
        if not book:
            raise ValueError(f"Livro com ID {loan.book_id} não encontrado")

        # Processa a devolução usando o serviço de domínio
        self._loan_service.return_loan(loan, book)

        # Salva as alterações
        updated_loan = await self._loan_repository.save(loan)
        await self._book_repository.save(book)

        return updated_loan

class RenewLoanUseCase:
    """Caso de uso: Renovar empréstimo"""

    def __init__(
        self,
        loan_repository: LoanRepositoryPort,
        notification_service: NotificationPort
    ):
        self._loan_repository = loan_repository
        self._notification_service = notification_service

    async def execute(self, loan_id: UUID) -> Loan:
        """Executa a renovação de um empréstimo"""

        loan = await self._loan_repository.find_by_id(loan_id)
        if not loan:
            raise ValueError(f"Empréstimo com ID {loan_id} não encontrado")

        loan.renew()

        return await self._loan_repository.save(loan)

class NotifyDueLoansUseCase:
    """Caso de uso: Notificar empréstimos vencendo"""

    def __init__(
        self,
        loan_repository: LoanRepositoryPort,
        user_repository: UserRepositoryPort,
        book_repository: BookRepositoryPort,
        notification_service: NotificationPort
    ):
        self._loan_repository = loan_repository
        self._user_repository = user_repository
        self._book_repository = book_repository
        self._notification_service = notification_service
        self._loan_service = LoanDomainService(loan_repository)

    async def execute(self) -> int:
        """Notifica usuários sobre empréstimos vencendo"""

        due_loans = self._loan_service.get_loans_due_soon()
        notifications_sent = 0

        for loan in due_loans:
            user = await self._user_repository.find_by_id(loan.user_id)
            book = await self._book_repository.find_by_id(loan.book_id)

            if user and book:
                success = await self._notification_service.send_notification(
                    recipient_email=user.email.value,
                    notification_type=NotificationType.DUE_REMINDER,
                    context={
                        "user_name": user.name,
                        "book_title": book.title,
                        "due_date": loan.due_date.strftime("%d/%m/%Y"),
                        "days_remaining": loan.days_until_due()
                    }
                )

                if success:
                    notifications_sent += 1

        return notifications_sent
```

### 3. Camada de Infraestrutura (Adapters)

#### `infrastructure/persistence/sqlalchemy_repositories.py`
```python
from typing import List, Optional
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from domain.entities.book import Book
from domain.entities.user import User
from domain.entities.loan import Loan
from application.ports.repositories import BookRepositoryPort, UserRepositoryPort, LoanRepositoryPort
from .models import BookModel, UserModel, LoanModel

class SQLAlchemyBookRepository(BookRepositoryPort):
    """Implementação do repositório de livros com SQLAlchemy"""

    def __init__(self, session: Session):
        self._session = session

    async def save(self, book: Book) -> Book:
        """Salva um livro no banco"""
        book_model = self._session.query(BookModel).filter_by(id=book.id).first()

        if book_model:
            # Atualiza existente
            book_model.title = book.title
            book_model.author = book.author
            book_model.isbn = book.isbn.value
            book_model.total_copies = book.total_copies
            book_model.available_copies = book.available_copies
            book_model.updated_at = book.updated_at
        else:
            # Cria novo
            book_model = BookModel(
                id=book.id,
                title=book.title,
                author=book.author,
                isbn=book.isbn.value,
                total_copies=book.total_copies,
                available_copies=book.available_copies,
                created_at=book.created_at,
                updated_at=book.updated_at
            )
            self._session.add(book_model)

        self._session.commit()
        return self._to_entity(book_model)

    async def find_by_id(self, book_id: UUID) -> Optional[Book]:
        """Busca livro por ID"""
        book_model = self._session.query(BookModel).filter_by(id=book_id).first()
        return self._to_entity(book_model) if book_model else None

    async def find_by_isbn(self, isbn: str) -> Optional[Book]:
        """Busca livro por ISBN"""
        book_model = self._session.query(BookModel).filter_by(isbn=isbn).first()
        return self._to_entity(book_model) if book_model else None

    async def find_by_title(self, title: str) -> List[Book]:
        """Busca livros por título (busca parcial)"""
        book_models = self._session.query(BookModel).filter(
            BookModel.title.ilike(f"%{title}%")
        ).all()
        return [self._to_entity(model) for model in book_models]

    async def find_all(self, limit: int = 100, offset: int = 0) -> List[Book]:
        """Lista todos os livros com paginação"""
        book_models = self._session.query(BookModel).offset(offset).limit(limit).all()
        return [self._to_entity(model) for model in book_models]

    async def delete(self, book_id: UUID) -> bool:
        """Remove um livro"""
        book_model = self._session.query(BookModel).filter_by(id=book_id).first()
        if book_model:
            self._session.delete(book_model)
            self._session.commit()
            return True
        return False

    def _to_entity(self, model: BookModel) -> Book:
        """Converte modelo para entidade"""
        from domain.value_objects.isbn import ISBN

        return Book(
            id=model.id,
            title=model.title,
            author=model.author,
            isbn=ISBN(model.isbn),
            total_copies=model.total_copies,
            available_copies=model.available_copies,
            created_at=model.created_at,
            updated_at=model.updated_at
        )

class SQLAlchemyLoanRepository(LoanRepositoryPort):
    """Implementação do repositório de empréstimos com SQLAlchemy"""

    def __init__(self, session: Session):
        self._session = session

    async def save(self, loan: Loan) -> Loan:
        """Salva um empréstimo no banco"""
        loan_model = self._session.query(LoanModel).filter_by(id=loan.id).first()

        if loan_model:
            # Atualiza existente
            loan_model.due_date = loan.due_date
            loan_model.return_date = loan.return_date
            loan_model.renewed_count = loan.renewed_count
        else:
            # Cria novo
            loan_model = LoanModel(
                id=loan.id,
                user_id=loan.user_id,
                book_id=loan.book_id,
                loan_date=loan.loan_date,
                due_date=loan.due_date,
                return_date=loan.return_date,
                renewed_count=loan.renewed_count,
                max_renewals=loan.max_renewals
            )
            self._session.add(loan_model)

        self._session.commit()
        return self._to_entity(loan_model)

    async def count_active_loans_by_user(self, user_id: UUID) -> int:
        """Conta empréstimos ativos do usuário"""
        return self._session.query(LoanModel).filter(
            and_(
                LoanModel.user_id == user_id,
                LoanModel.return_date.is_(None)
            )
        ).count()

    async def user_has_active_loan_for_book(self, user_id: UUID, book_id: UUID) -> bool:
        """Verifica se usuário tem empréstimo ativo do livro"""
        loan = self._session.query(LoanModel).filter(
            and_(
                LoanModel.user_id == user_id,
                LoanModel.book_id == book_id,
                LoanModel.return_date.is_(None)
            )
        ).first()
        return loan is not None

    async def find_loans_due_in_days(self, days: int) -> List[Loan]:
        """Busca empréstimos que vencem em X dias"""
        from datetime import datetime, timedelta

        target_date = datetime.utcnow() + timedelta(days=days)
        start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)

        loan_models = self._session.query(LoanModel).filter(
            and_(
                LoanModel.return_date.is_(None),
                LoanModel.due_date >= start_of_day,
                LoanModel.due_date <= end_of_day
            )
        ).all()

        return [self._to_entity(model) for model in loan_models]

    async def find_overdue_loans(self) -> List[Loan]:
        """Busca empréstimos vencidos"""
        from datetime import datetime

        loan_models = self._session.query(LoanModel).filter(
            and_(
                LoanModel.return_date.is_(None),
                LoanModel.due_date < datetime.utcnow()
            )
        ).all()

        return [self._to_entity(model) for model in loan_models]

    def _to_entity(self, model: LoanModel) -> Loan:
        """Converte modelo para entidade"""
        return Loan(
            id=model.id,
            user_id=model.user_id,
            book_id=model.book_id,
            loan_date=model.loan_date,
            due_date=model.due_date,
            return_date=model.return_date,
            renewed_count=model.renewed_count,
            max_renewals=model.max_renewals
        )
```

#### `infrastructure/persistence/models.py`
```python
from sqlalchemy import Column, String, Integer, DateTime, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import uuid

Base = declarative_base()

class BookModel(Base):
    """Modelo SQLAlchemy para livros"""
    __tablename__ = "books"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False, index=True)
    author = Column(String(255), nullable=False, index=True)
    isbn = Column(String(13), nullable=False, unique=True, index=True)
    total_copies = Column(Integer, nullable=False, default=0)
    available_copies = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

    # Relacionamentos
    loans = relationship("LoanModel", back_populates="book")

class UserModel(Base):
    """Modelo SQLAlchemy para usuários"""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    max_borrowed_books = Column(Integer, nullable=False, default=3)
    created_at = Column(DateTime, nullable=False)

    # Relacionamentos
    loans = relationship("LoanModel", back_populates="user")

class LoanModel(Base):
    """Modelo SQLAlchemy para empréstimos"""
    __tablename__ = "loans"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    book_id = Column(UUID(as_uuid=True), ForeignKey("books.id"), nullable=False, index=True)
    loan_date = Column(DateTime, nullable=False, index=True)
    due_date = Column(DateTime, nullable=False, index=True)
    return_date = Column(DateTime, nullable=True, index=True)
    renewed_count = Column(Integer, nullable=False, default=0)
    max_renewals = Column(Integer, nullable=False, default=2)

    # Relacionamentos
    user = relationship("UserModel", back_populates="loans")
    book = relationship("BookModel", back_populates="loans")
```

#### `infrastructure/web/fastapi_controllers.py`
```python
from fastapi import APIRouter, HTTPException, Depends, status
from typing import List
from uuid import UUID

from application.use_cases.book_management import CreateBookUseCase, FindBookUseCase, UpdateBookCopiesUseCase
from application.use_cases.loan_management import BorrowBookUseCase, ReturnBookUseCase, RenewLoanUseCase
from .schemas import BookCreateRequest, BookResponse, LoanResponse, BorrowBookRequest
from .dependencies import get_book_use_cases, get_loan_use_cases

router = APIRouter()

# Endpoints de Livros
@router.post("/books", response_model=BookResponse, status_code=status.HTTP_201_CREATED)
async def create_book(
    request: BookCreateRequest,
    use_cases = Depends(get_book_use_cases)
):
    """Cria um novo livro"""
    try:
        create_use_case: CreateBookUseCase = use_cases["create"]
        book = await create_use_case.execute(
            title=request.title,
            author=request.author,
            isbn=request.isbn,
            total_copies=request.total_copies
        )
        return BookResponse.from_entity(book)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/books/{book_id}", response_model=BookResponse)
async def get_book(
    book_id: UUID,
    use_cases = Depends(get_book_use_cases)
):
    """Busca livro por ID"""
    find_use_case: FindBookUseCase = use_cases["find"]
    book = await find_use_case.execute_by_id(book_id)

    if not book:
        raise HTTPException(status_code=404, detail="Livro não encontrado")

    return BookResponse.from_entity(book)

@router.get("/books", response_model=List[BookResponse])
async def search_books(
    title: str = None,
    isbn: str = None,
    use_cases = Depends(get_book_use_cases)
):
    """Busca livros por título ou ISBN"""
    find_use_case: FindBookUseCase = use_cases["find"]

    if isbn:
        book = await find_use_case.execute_by_isbn(isbn)
        return [BookResponse.from_entity(book)] if book else []
    elif title:
        books = await find_use_case.execute_by_title(title)
        return [BookResponse.from_entity(book) for book in books]
    else:
        raise HTTPException(status_code=400, detail="Informe título ou ISBN para busca")

@router.patch("/books/{book_id}/copies")
async def add_book_copies(
    book_id: UUID,
    additional_copies: int,
    use_cases = Depends(get_book_use_cases)
):
    """Adiciona cópias ao acervo"""
    try:
        update_use_case: UpdateBookCopiesUseCase = use_cases["update_copies"]
        book = await update_use_case.execute(book_id, additional_copies)
        return BookResponse.from_entity(book)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoints de Empréstimos
@router.post("/loans", response_model=LoanResponse, status_code=status.HTTP_201_CREATED)
async def borrow_book(
    request: BorrowBookRequest,
    use_cases = Depends(get_loan_use_cases)
):
    """Empresta um livro"""
    try:
        borrow_use_case: BorrowBookUseCase = use_cases["borrow"]
        loan = await borrow_use_case.execute(request.user_id, request.book_id)
        return LoanResponse.from_entity(loan)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.patch("/loans/{loan_id}/return")
async def return_book(
    loan_id: UUID,
    use_cases = Depends(get_loan_use_cases)
):
    """Devolve um livro"""
    try:
        return_use_case: ReturnBookUseCase = use_cases["return"]
        loan = await return_use_case.execute(loan_id)
        return LoanResponse.from_entity(loan)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.patch("/loans/{loan_id}/renew")
async def renew_loan(
    loan_id: UUID,
    use_cases = Depends(get_loan_use_cases)
):
    """Renova um empréstimo"""
    try:
        renew_use_case: RenewLoanUseCase = use_cases["renew"]
        loan = await renew_use_case.execute(loan_id)
        return LoanResponse.from_entity(loan)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

#### `infrastructure/web/schemas.py`
```python
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime
from typing import Optional
from uuid import UUID

from domain.entities.book import Book
from domain.entities.loan import Loan
from domain.entities.user import User

class BookCreateRequest(BaseModel):
    """Schema para criação de livro"""
    title: str = Field(..., min_length=1, max_length=255)
    author: str = Field(..., min_length=1, max_length=255)
    isbn: str = Field(..., min_length=10, max_length=13)
    total_copies: int = Field(..., ge=1)

class BookResponse(BaseModel):
    """Schema de resposta para livro"""
    id: UUID
    title: str
    author: str
    isbn: str
    total_copies: int
    available_copies: int
    status: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_entity(cls, book: Book) -> "BookResponse":
        return cls(
            id=book.id,
            title=book.title,
            author=book.author,
            isbn=book.isbn.value,
            total_copies=book.total_copies,
            available_copies=book.available_copies,
            status=book.get_status().value,
            created_at=book.created_at,
            updated_at=book.updated_at
        )

class UserCreateRequest(BaseModel):
    """Schema para criação de usuário"""
    name: str = Field(..., min_length=1, max_length=255)
    email: EmailStr
    max_borrowed_books: int = Field(3, ge=1, le=10)

class UserResponse(BaseModel):
    """Schema de resposta para usuário"""
    id: UUID
    name: str
    email: str
    max_borrowed_books: int
    created_at: datetime

    @classmethod
    def from_entity(cls, user: User) -> "UserResponse":
        return cls(
            id=user.id,
            name=user.name,
            email=user.email.value,
            max_borrowed_books=user.max_borrowed_books,
            created_at=user.created_at
        )

class BorrowBookRequest(BaseModel):
    """Schema para empréstimo de livro"""
    user_id: UUID
    book_id: UUID

class LoanResponse(BaseModel):
    """Schema de resposta para empréstimo"""
    id: UUID
    user_id: UUID
    book_id: UUID
    loan_date: datetime
    due_date: datetime
    return_date: Optional[datetime]
    renewed_count: int
    max_renewals: int
    status: str
    days_until_due: int

    @classmethod
    def from_entity(cls, loan: Loan) -> "LoanResponse":
        return cls(
            id=loan.id,
            user_id=loan.user_id,
            book_id=loan.book_id,
            loan_date=loan.loan_date,
            due_date=loan.due_date,
            return_date=loan.return_date,
            renewed_count=loan.renewed_count,
            max_renewals=loan.max_renewals,
            status=loan.get_status().value,
            days_until_due=loan.days_until_due()
        )
```

#### `infrastructure/notifications/email_adapter.py`
```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any

from application.ports.notifications import NotificationPort, EmailServicePort
from domain.value_objects.enums import NotificationType

class SMTPEmailService(EmailServicePort):
    """Implementação do serviço de email via SMTP"""

    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password

    async def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        html_body: str = None
    ) -> bool:
        """Envia email via SMTP"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.username
            msg['To'] = to

            # Adiciona corpo em texto
            text_part = MIMEText(body, 'plain', 'utf-8')
            msg.attach(text_part)

            # Adiciona corpo em HTML se fornecido
            if html_body:
                html_part = MIMEText(html_body, 'html', 'utf-8')
                msg.attach(html_part)

            # Conecta e envia
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            return True

        except Exception as e:
            print(f"Erro ao enviar email: {e}")
            return False

class LibraryNotificationService(NotificationPort):
    """Serviço de notificações da biblioteca"""

    def __init__(self, email_service: EmailServicePort):
        self._email_service = email_service

    async def send_notification(
        self,
        recipient_email: str,
        notification_type: NotificationType,
        context: Dict[str, Any]
    ) -> bool:
        """Envia notificação baseada no tipo"""

        template = self._get_template(notification_type)
        if not template:
            return False

        subject = template["subject"].format(**context)
        body = template["body"].format(**context)
        html_body = template.get("html_body", "").format(**context)

        return await self._email_service.send_email(
            to=recipient_email,
            subject=subject,
            body=body,
            html_body=html_body if html_body else None
        )

    def _get_template(self, notification_type: NotificationType) -> Dict[str, str]:
        """Retorna template da notificação"""
        templates = {
            NotificationType.DUE_REMINDER: {
                "subject": "Lembrete: Livro vence em {days_remaining} dias",
                "body": """Olá {user_name},

Este é um lembrete de que o livro "{book_title}" deve ser devolvido em {days_remaining} dias.

Data de vencimento: {due_date}

Por favor, devolva o livro na data correta para evitar multas.

Biblioteca Digital""",
                "html_body": """
                <h2>Lembrete de Devolução</h2>
                <p>Olá <strong>{user_name}</strong>,</p>
                <p>Este é um lembrete de que o livro <em>"{book_title}"</em> deve ser devolvido em <strong>{days_remaining} dias</strong>.</p>
                <p><strong>Data de vencimento:</strong> {due_date}</p>
                <p>Por favor, devolva o livro na data correta para evitar multas.</p>
                <hr>
                <p><em>Biblioteca Digital</em></p>
                """
            },
            NotificationType.OVERDUE_NOTICE: {
                "subject": "URGENTE: Livro em atraso",
                "body": """Olá {user_name},

O livro "{book_title}" está em atraso desde {due_date}.

Por favor, devolva o livro o mais rápido possível. Multas podem ser aplicadas.

Biblioteca Digital""",
                "html_body": """
                <h2 style="color: red;">AVISO: Livro em Atraso</h2>
                <p>Olá <strong>{user_name}</strong>,</p>
                <p>O livro <em>"{book_title}"</em> está <strong style="color: red;">em atraso</strong> desde {due_date}.</p>
                <p>Por favor, devolva o livro o mais rápido possível. Multas podem ser aplicadas.</p>
                <hr>
                <p><em>Biblioteca Digital</em></p>
                """
            },
            NotificationType.RETURN_CONFIRMATION: {
                "subject": "Empréstimo realizado com sucesso",
                "body": """Olá {user_name},

O empréstimo do livro "{book_title}" foi realizado com sucesso!

Data de vencimento: {due_date}

Aproveite a leitura!

Biblioteca Digital""",
                "html_body": """
                <h2 style="color: green;">Empréstimo Confirmado</h2>
                <p>Olá <strong>{user_name}</strong>,</p>
                <p>O empréstimo do livro <em>"{book_title}"</em> foi realizado com sucesso!</p>
                <p><strong>Data de vencimento:</strong> {due_date}</p>
                <p>Aproveite a leitura!</p>
                <hr>
                <p><em>Biblioteca Digital</em></p>
                """
            }
        }

        return templates.get(notification_type, {})
```

### 4. Configuração e Dependency Injection

#### `infrastructure/config/dependencies.py`
```python
from functools import lru_cache
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from application.use_cases.book_management import CreateBookUseCase, FindBookUseCase, UpdateBookCopiesUseCase
from application.use_cases.loan_management import BorrowBookUseCase, ReturnBookUseCase, RenewLoanUseCase
from infrastructure.persistence.sqlalchemy_repositories import SQLAlchemyBookRepository, SQLAlchemyLoanRepository, SQLAlchemyUserRepository
from infrastructure.notifications.email_adapter import SMTPEmailService, LibraryNotificationService
from .settings import get_settings

# Configuração do banco de dados
@lru_cache()
def get_database_engine():
    settings = get_settings()
    engine = create_engine(settings.database_url)
    return engine

@lru_cache()
def get_session_factory():
    engine = get_database_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_database_session() -> Session:
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

# Repositórios
def get_book_repository(session: Session = Depends(get_database_session)):
    return SQLAlchemyBookRepository(session)

def get_user_repository(session: Session = Depends(get_database_session)):
    return SQLAlchemyUserRepository(session)

def get_loan_repository(session: Session = Depends(get_database_session)):
    return SQLAlchemyLoanRepository(session)

# Serviços
@lru_cache()
def get_email_service():
    settings = get_settings()
    return SMTPEmailService(
        smtp_host=settings.smtp_host,
        smtp_port=settings.smtp_port,
        username=settings.smtp_username,
        password=settings.smtp_password
    )

def get_notification_service():
    email_service = get_email_service()
    return LibraryNotificationService(email_service)

# Use Cases
def get_book_use_cases(
    book_repository = Depends(get_book_repository)
):
    return {
        "create": CreateBookUseCase(book_repository),
        "find": FindBookUseCase(book_repository),
        "update_copies": UpdateBookCopiesUseCase(book_repository)
    }

def get_loan_use_cases(
    book_repository = Depends(get_book_repository),
    user_repository = Depends(get_user_repository),
    loan_repository = Depends(get_loan_repository),
    notification_service = Depends(get_notification_service)
):
    return {
        "borrow": BorrowBookUseCase(book_repository, user_repository, loan_repository, notification_service),
        "return": ReturnBookUseCase(book_repository, loan_repository, notification_service),
        "renew": RenewLoanUseCase(loan_repository, notification_service)
    }
```

#### `infrastructure/config/settings.py`
```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Configurações da aplicação"""

    # Database
    database_url: str = "postgresql://user:password@localhost/library_db"

    # SMTP Settings
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""

    # Application
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
```

### 5. Aplicação FastAPI

#### `main.py`
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from infrastructure.web.fastapi_controllers import router
from infrastructure.config.settings import get_settings

def create_app() -> FastAPI:
    """Factory para criar a aplicação FastAPI"""

    settings = get_settings()

    app = FastAPI(
        title="Sistema de Biblioteca Digital",
        description="API para gerenciamento de biblioteca com arquitetura hexagonal",
        version="1.0.0",
        debug=settings.debug
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(router, prefix="/api/v1")

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "message": "Library API is running"}

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
```

### 6. Testes

#### `tests/unit/test_book_entity.py`
```python
import pytest
from datetime import datetime
from uuid import uuid4

from domain.entities.book import Book
from domain.value_objects.isbn import ISBN

class TestBookEntity:
    """Testes unitários para entidade Book"""

    def test_create_valid_book(self):
        """Teste: criar livro válido"""
        book = Book(
            title="Clean Architecture",
            author="Robert C. Martin",
            isbn=ISBN("9780134494166"),
            total_copies=5,
            available_copies=5
        )

        assert book.title == "Clean Architecture"
        assert book.author == "Robert C. Martin"
        assert book.isbn.value == "9780134494166"
        assert book.total_copies == 5
        assert book.available_copies == 5
        assert book.is_available() is True

    def test_book_without_title_raises_error(self):
        """Teste: livro sem título deve gerar erro"""
        with pytest.raises(ValueError, match="Título é obrigatório"):
            Book(
                title="",
                author="Robert C. Martin",
                isbn=ISBN("9780134494166"),
                total_copies=5,
                available_copies=5
            )

    def test_borrow_copy_decreases_availability(self):
        """Teste: emprestar cópia diminui disponibilidade"""
        book = Book(
            title="Clean Code",
            author="Robert C. Martin",
            isbn=ISBN("9780132350884"),
            total_copies=3,
            available_copies=3
        )

        book.borrow_copy()

        assert book.available_copies == 2
        assert book.total_copies == 3
        assert book.is_available() is True

    def test_borrow_last_copy_makes_unavailable(self):
        """Teste: emprestar última cópia torna indisponível"""
        book = Book(
            title="The Pragmatic Programmer",
            author="David Thomas",
            isbn=ISBN("9780201616224"),
            total_copies=1,
            available_copies=1
        )

        book.borrow_copy()

        assert book.available_copies == 0
        assert book.is_available() is False

    def test_borrow_unavailable_book_raises_error(self):
        """Teste: emprestar livro indisponível deve gerar erro"""
        book = Book(
            title="Design Patterns",
            author="Gang of Four",
            isbn=ISBN("9780201633610"),
            total_copies=2,
            available_copies=0
        )

        with pytest.raises(ValueError, match="não possui cópias disponíveis"):
            book.borrow_copy()

    def test_return_copy_increases_availability(self):
        """Teste: devolver cópia aumenta disponibilidade"""
        book = Book(
            title="Refactoring",
            author="Martin Fowler",
            isbn=ISBN("9780201485677"),
            total_copies=3,
            available_copies=1
        )

        book.return_copy()

        assert book.available_copies == 2
        assert book.total_copies == 3

    def test_add_copies_increases_total_and_available(self):
        """Teste: adicionar cópias aumenta total e disponível"""
        book = Book(
            title="Domain-Driven Design",
            author="Eric Evans",
            isbn=ISBN("9780321125217"),
            total_copies=2,
            available_copies=1
        )

        book.add_copies(3)

        assert book.total_copies == 5
        assert book.available_copies == 4
```

#### `tests/integration/test_book_use_cases.py`
```python
import pytest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from application.use_cases.book_management import CreateBookUseCase, FindBookUseCase
from domain.entities.book import Book
from domain.value_objects.isbn import ISBN

class TestBookUseCases:
    """Testes de integração para casos de uso de livros"""

    @pytest.fixture
    def mock_book_repository(self):
        """Mock do repositório de livros"""
        return AsyncMock()

    @pytest.fixture
    def create_book_use_case(self, mock_book_repository):
        """Instância do caso de uso de criação"""
        return CreateBookUseCase(mock_book_repository)

    @pytest.fixture
    def find_book_use_case(self, mock_book_repository):
        """Instância do caso de uso de busca"""
        return FindBookUseCase(mock_book_repository)

    @pytest.mark.asyncio
    async def test_create_book_success(self, create_book_use_case, mock_book_repository):
        """Teste: criar livro com sucesso"""
        # Arrange
        mock_book_repository.find_by_isbn.return_value = None  # ISBN não existe
        mock_book_repository.save.return_value = Book(
            id=uuid4(),
            title="Test Book",
            author="Test Author",
            isbn=ISBN("9780134494166"),
            total_copies=5,
            available_copies=5
        )

        # Act
        result = await create_book_use_case.execute(
            title="Test Book",
            author="Test Author",
            isbn="9780134494166",
            total_copies=5
        )

        # Assert
        assert result.title == "Test Book"
        assert result.author == "Test Author"
        assert result.isbn.value == "9780134494166"
        mock_book_repository.find_by_isbn.assert_called_once_with("9780134494166")
        mock_book_repository.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_book_duplicate_isbn_fails(self, create_book_use_case, mock_book_repository):
        """Teste: criar livro com ISBN duplicado deve falhar"""
        # Arrange
        existing_book = Book(
            title="Existing Book",
            author="Existing Author",
            isbn=ISBN("9780134494166"),
            total_copies=3,
            available_copies=3
        )
        mock_book_repository.find_by_isbn.return_value = existing_book

        # Act & Assert
        with pytest.raises(ValueError, match="Já existe um livro com ISBN"):
            await create_book_use_case.execute(
                title="New Book",
                author="New Author",
                isbn="9780134494166",
                total_copies=5
            )

        mock_book_repository.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_find_book_by_id_success(self, find_book_use_case, mock_book_repository):
        """Teste: buscar livro por ID com sucesso"""
        # Arrange
        book_id = uuid4()
        expected_book = Book(
            id=book_id,
            title="Found Book",
            author="Found Author",
            isbn=ISBN("9780134494166"),
            total_copies=5,
            available_copies=3
        )
        mock_book_repository.find_by_id.return_value = expected_book

        # Act
        result = await find_book_use_case.execute_by_id(book_id)

        # Assert
        assert result == expected_book
        mock_book_repository.find_by_id.assert_called_once_with(book_id)

    @pytest.mark.asyncio
    async def test_find_book_by_id_not_found(self, find_book_use_case, mock_book_repository):
        """Teste: buscar livro inexistente retorna None"""
        # Arrange
        book_id = uuid4()
        mock_book_repository.find_by_id.return_value = None

        # Act
        result = await find_book_use_case.execute_by_id(book_id)

        # Assert
        assert result is None
        mock_book_repository.find_by_id.assert_called_once_with(book_id)
```

## 🎯 Benefícios da Arquitetura Hexagonal Demonstrados

### 1. **Testabilidade**
- **Domínio isolado**: Entidades e value objects podem ser testados sem dependências externas
- **Mocks simples**: Interfaces permitem mocks fáceis nos testes
- **Diferentes níveis**: Testes unitários, de integração e E2E independentes

### 2. **Flexibilidade**
- **Troca de banco**: Podemos trocar PostgreSQL por MongoDB sem afetar o domínio
- **Troca de framework**: FastAPI pode ser substituído por Flask sem mudanças no core
- **Múltiplos adapters**: Web API, CLI, desktop app usando a mesma lógica

### 3. **Manutenibilidade**
- **Separação clara**: Cada camada tem responsabilidade bem definida
- **Baixo acoplamento**: Mudanças em uma camada não afetam outras
- **Alto coesão**: Código relacionado está agrupado

### 4. **Extensibilidade**
- **Novos use cases**: Fácil adição de novas funcionalidades
- **Novos adapters**: Simples implementação de novos pontos de entrada/saída
- **Novas regras**: Lógica de negócio centralizada e fácil de modificar

## 🚀 Como Executar o Exemplo

### 1. Instalar Dependências
```bash
pip install fastapi uvicorn sqlalchemy psycopg2-binary alembic pytest pytest-asyncio
```

### 2. Configurar Banco de Dados
```bash
# PostgreSQL
createdb library_db

# Executar migrações (usando Alembic)
alembic upgrade head
```

### 3. Configurar Variáveis de Ambiente
```bash
export DATABASE_URL="postgresql://user:password@localhost/library_db"
export SMTP_HOST="smtp.gmail.com"
export SMTP_USERNAME="your-email@gmail.com"
export SMTP_PASSWORD="your-app-password"
```

### 4. Executar Aplicação
```bash
python main.py
```

### 5. Executar Testes
```bash
pytest tests/ -v
```

Esta implementação demonstra na prática todos os conceitos da arquitetura hexagonal com um exemplo real e completo que pode ser executado e testado.