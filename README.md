# RAG Pipeline Backend

A sophisticated Retrieval-Augmented Generation (RAG) pipeline backend system that combines advanced language models with knowledge retrieval capabilities, specifically designed for medical knowledge management and response generation. The system implements a graph-based knowledge representation with vector embeddings for efficient semantic search and retrieval.

## Project Overview

This project implements a backend system that leverages RAG (Retrieval-Augmented Generation) to provide intelligent responses to medical queries. It combines the power of large language models with a graph-based knowledge base to generate accurate and contextually relevant responses, with a focus on respiratory diseases. The system uses a multi-stage pipeline that includes semantic search, graph-based knowledge retrieval, and context-aware response generation.

## Core Features

### 1. User Interaction System
- Question answering through RAG pipeline
  * Semantic understanding of medical queries
  * Context-aware response generation
  * Multi-turn conversation support
  * Real-time response streaming
- Conversation history management
  * Persistent storage of all interactions
  * Chronological ordering of Q&A pairs
  * Context preservation across sessions
- Response validation and feedback system
  * Doctor-led validation workflow
  * Correction and improvement tracking
  * Quality metrics monitoring
- User-specific conversation tracking
  * Individual conversation threads
  * User-specific context management
  * Privacy-preserving data storage
- Multi-turn conversation support
  * Context retention across turns
  * Follow-up question handling
  * Conversation state management
- Response validation workflow with doctor oversight
  * Expert review of AI-generated responses
  * Medical accuracy verification
  * Correction and improvement suggestions

### 2. RAG Pipeline Components
- Knowledge retrieval system using PCST (Prize-Collecting Steiner Tree) algorithm
  * Graph-based knowledge representation
  * Semantic similarity scoring
  * Subgraph extraction for context
  * Edge and node relevance weighting
- Response generation using NVIDIA's LLM API
  * Context-aware text generation
  * Medical terminology handling
  * Uncertainty expression
  * Structured response formatting
- Sentence embedding models for semantic search
  * Transformer-based embeddings
  * Multi-dimensional vector representation
  * Cosine similarity matching
  * Batch processing support
- Vector database integration with Milvus
  * Efficient similarity search
  * Scalable vector storage
  * Real-time indexing
  * Batch query support
- Graph-based knowledge representation
  * Node-edge relationship modeling
  * Semantic graph construction
  * Dynamic graph updates
  * Subgraph extraction
- Automated evaluation system for response quality
  * Multi-criteria assessment
  * Quantitative scoring
  * Quality metrics tracking
  * Performance monitoring

### 3. Multi-User Support
- User management system with role-based access
  * Secure authentication
  * Permission management
  * Session handling
  * Activity logging
- Three distinct user roles:
  * Users: Submit questions and receive responses
    - Question submission
    - Response viewing
    - Conversation history access
  * Doctors: Validate and correct AI-generated responses
    - Response validation
    - Medical accuracy verification
    - Correction submission
    - Quality assessment
  * Admins: Manage knowledge base and system settings
    - System configuration
    - Knowledge base management
    - User administration
    - Performance monitoring
- Separate conversation tracking per user
  * Individual conversation threads
  * Privacy-preserving storage
  * Context management
- Response validation workflow with doctor oversight
  * Expert review process
  * Quality assurance
  * Feedback integration
  * Performance monitoring

### 4. Knowledge Base Management
- Vector-based knowledge storage
  * Efficient similarity search
  * Scalable storage system
  * Real-time updates
  * Batch processing support
- Semantic search capabilities
  * Context-aware retrieval
  * Relevance scoring
  * Query understanding
  * Result ranking
- Knowledge graph integration
  * Graph-based representation
  * Relationship modeling
  * Dynamic updates
  * Subgraph extraction
- Efficient retrieval mechanisms
  * Optimized search algorithms
  * Caching strategies
  * Batch processing
  * Real-time updates
- Automated data processing pipeline
  * Data collection
  * Information extraction
  * Knowledge graph construction
  * Vector embedding generation
- Support for multiple medical sources:
  * World Health Organization (WHO)
    - Disease fact sheets
    - Treatment guidelines
    - Prevention information
  * Centers for Disease Control (CDC)
    - Disease statistics
    - Public health guidelines
    - Research findings
  * National Institutes of Health (NIH)
    - Medical research
    - Clinical trials
    - Treatment protocols

## Technical Architecture

### Backend Framework
- Flask-based REST API
  * RESTful endpoint design
  * Request validation
  * Response formatting
  * Error handling
- SQLAlchemy ORM for database management
  * Object-relational mapping
  * Query optimization
  * Transaction management
  * Schema management
- CORS support for cross-origin requests
  * Security configuration
  * Access control
  * Request validation
- Modular blueprint architecture
  * Component isolation
  * Code organization
  * Maintainability
  * Scalability
- Streaming response support
  * Real-time data transfer
  * Chunked responses
  * Progress tracking
- Error handling and logging
  * Comprehensive error tracking
  * Debug information
  * Performance monitoring
  * System health checks

### Core Technologies
- PyTorch and PyTorch Geometric for ML operations
  * Deep learning models
  * Graph neural networks
  * Tensor operations
  * GPU acceleration
- Transformers library for language models
  * Pre-trained models
  * Fine-tuning support
  * Tokenization
  * Model inference
- Milvus for vector search
  * Vector similarity search
  * Index management
  * Query optimization
  * Scalability
- LangChain for RAG pipeline orchestration
  * Pipeline management
  * Component integration
  * Workflow automation
  * State management
- BeautifulSoup4 for web scraping
  * HTML parsing
  * Content extraction
  * Data cleaning
  * Structure preservation
- NetworkX for graph operations
  * Graph manipulation
  * Path finding
  * Component analysis
  * Graph algorithms

### Database Structure
- User management
  * User profiles
  * Authentication data
  * Role assignments
  * Activity tracking
- Conversation tracking
  * Message history
  * Context preservation
  * Timestamp tracking
  * State management
- Message storage
  * Text content
  * Metadata
  * Relationships
  * Timestamps
- Response management
  * Generated responses
  * Validation status
  * Corrections
  * Quality metrics
- Validation tracking
  * Expert reviews
  * Corrections
  * Quality scores
  * Feedback
- Knowledge graph storage
  * Node data
  * Edge relationships
  * Metadata
  * Timestamps
- Vector embeddings
  * Sentence embeddings
  * Graph embeddings
  * Update tracking
  * Version control

## API Endpoints

### User Routes
- `/api/user/ask` - Submit questions and receive AI-generated responses
  * Validates user existence
  * Creates/retrieves conversation
  * Runs RAG pipeline
  * Saves generated response
  * Handles streaming responses
  * Manages conversation context
- `/api/user/conversation` - Retrieve conversation history
  * Returns ordered Q&A pairs
  * Includes timestamps
  * Manages pagination
  * Filters by date range
  * Includes validation status
- `/api/user/inbox` - Access validated responses
  * Shows doctor-validated responses
  * Includes corrections if any
  * Provides quality metrics
  * Tracks validation history
  * Manages response status

### Doctor Routes
- `/api/doctor/inbox` - View pending responses
  * Lists responses awaiting validation
  * Includes original questions and generated answers
  * Shows conversation context
  * Provides validation interface
  * Tracks validation status
- `/api/doctor/validate/<response_id>` - Validate responses
  * Accept or reject responses
  * Provide corrections if needed
  * Update response status
  * Add quality metrics
  * Track validation history

### Admin Routes
- `/api/admin/upload` - Upload new knowledge base data
  * Handles file uploads
  * Processes new data
  * Updates knowledge base
  * Manages version control
- `/api/admin/settings` - Manage system settings
  * Configuration management
  * System parameters
  * Feature toggles
  * Performance settings
- `/api/admin/evaluate` - Trigger evaluation process
  * Quality assessment
  * Performance metrics
  * System health checks
  * Usage statistics
- `/api/admin/dashboard` - View system overview
  * System metrics
  * Usage statistics
  * Performance data
  * Health monitoring
- `/api/admin/evaluation-report` - Access evaluation metrics
  * Quality scores
  * Performance data
  * Usage statistics
  * System health

## Knowledge Base Processing

### Data Collection
- Automated web scraping from medical sources
  * HTML parsing
  * Content extraction
  * Structure preservation
  * Data cleaning
- Support for multiple disease types:
  * Asthma
    - Symptoms
    - Treatments
    - Prevention
    - Research
  * COPD
    - Clinical guidelines
    - Management strategies
    - Risk factors
    - Treatment options
  * Pneumonia
    - Diagnosis criteria
    - Treatment protocols
    - Prevention methods
    - Risk assessment
  * Tuberculosis
    - Treatment guidelines
    - Drug resistance
    - Prevention strategies
    - Public health measures
  * COVID-19
    - Latest research
    - Treatment protocols
    - Prevention guidelines
    - Public health measures
- Structured data extraction
  * Entity recognition
  * Relationship extraction
  * Fact validation
  * Quality control
- JSON format storage
  * Structured data
  * Metadata preservation
  * Version control
  * Access management

### Data Processing Pipeline
1. Raw data collection and organization
   * Web scraping
   * File processing
   * Data validation
   * Structure preservation
2. SPO (Subject-Predicate-Object) triple extraction
   * Entity recognition
   * Relationship identification
   * Fact validation
   * Quality control
3. Normalization and deduplication
   * Term standardization
   * Duplicate removal
   * Consistency checking
   * Quality assurance
4. Refinement and validation
   * Expert review
   * Quality assessment
   * Error correction
   * Consistency verification
5. Graph construction and embedding
   * Node creation
   * Edge establishment
   * Graph validation
   * Vector generation

### Storage Structure
- `/raw_files`: Original data
  * Source documents
  * Metadata
  * Version history
  * Access logs
- `/spo`: Subject-Predicate-Object triples
  - `/raw`: Initial extractions
    * Unprocessed triples
    * Source references
    * Extraction metadata
  - `/normalized&deduplicated`: Cleaned data
    * Standardized terms
    * Unique triples
    * Quality metrics
  - `/refined`: Final validated triples
    * Expert-verified data
    * Quality scores
    * Validation history
- `/nodes`: Graph node definitions
  * Entity data
  * Properties
  * Relationships
  * Metadata
- `/edges`: Graph edge definitions
  * Relationship types
  * Properties
  * Weights
  * Metadata
- `/graphs`: Graph embeddings
  * Vector representations
  * Update history
  * Version control
- `/graphs_json`: Graph representations
  * Full graph data
  * Structure information
  * Metadata
  * Version history

## Response Generation

### Evaluation Criteria (0-5 scale)
- Clarity: Response structure and presentation
  * Organization
  * Readability
  * Terminology
  * Formatting
- Exactitude: Accuracy and precision
  * Factual correctness
  * Medical accuracy
  * Source reliability
  * Detail level
- Context Adherence: Alignment with knowledge graphs
  * Source consistency
  * Context relevance
  * Information completeness
  * Relationship accuracy
- Relevance: Context relevance to question
  * Query understanding
  * Answer focus
  * Information pertinence
  * Detail appropriateness
- Completeness: Thoroughness of response
  * Information coverage
  * Detail level
  * Source inclusion
  * Context provision
- Logical Flow: Coherence and structure
  * Argument organization
  * Transition clarity
  * Conclusion strength
  * Information hierarchy
- Uncertainty Handling: Acknowledgment of limitations
  * Confidence expression
  * Limitation disclosure
  * Alternative consideration
  * Source reliability

### Response Processing
1. Question embedding
   * Text preprocessing
   * Vector generation
   * Context analysis
   * Query understanding
2. Knowledge graph retrieval
   * Graph traversal
   * Relevance scoring
   * Context matching
   * Source validation
3. Subgraph extraction
   * PCST algorithm
   * Relevance weighting
   * Context preservation
   * Structure maintenance
4. Response generation
   * Context integration
   * Text generation
   * Formatting
   * Quality control
5. Post-processing and cleaning
   * Format standardization
   * Error correction
   * Quality verification
   * Structure validation
6. Optional evaluation
   * Quality assessment
   * Metric calculation
   * Performance tracking
   * Improvement identification
7. Doctor validation
   * Expert review
   * Accuracy verification
   * Correction application
   * Quality assurance

## Dependencies

### Core ML & Deep Learning
- PyTorch 2.1.0
  * Deep learning framework
  * GPU acceleration
  * Tensor operations
  * Model training
- PyTorch Geometric
  * Graph neural networks
  * Graph operations
  * Message passing
  * Graph algorithms
- Transformers 4.39.3
  * Language models
  * Tokenization
  * Model inference
  * Fine-tuning
- PEFT 0.10.0
  * Parameter efficient fine-tuning
  * Model optimization
  * Resource management
  * Performance tuning

### Vector Search & RAG
- Milvus
  * Vector similarity search
  * Index management
  * Query optimization
  * Scalability
- LangChain
  * RAG pipeline
  * Component integration
  * Workflow automation
  * State management
- OpenAI integration
  * Language model access
  * Text generation
  * Embedding generation
  * Model fine-tuning
- NVIDIA API integration
  * GPU acceleration
  * Model inference
  * Performance optimization
  * Resource management

### Web Framework
- Flask
  * Web server
  * Route handling
  * Request processing
  * Response generation
- Flask-SQLAlchemy
  * Database ORM
  * Query building
  * Transaction management
  * Schema handling
- Flask-CORS
  * Cross-origin support
  * Security configuration
  * Access control
  * Request validation

### Utilities
- Pandas
  * Data manipulation
  * Analysis
  * Processing
  * Storage
- NetworkX
  * Graph operations
  * Network analysis
  * Path finding
  * Component analysis
- BeautifulSoup4
  * HTML parsing
  * Content extraction
  * Data cleaning
  * Structure preservation
- SciPy
  * Scientific computing
  * Numerical operations
  * Statistical analysis
  * Optimization

## Setup and Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
- Set up database credentials
  * Connection strings
  * Authentication
  * Access control
  * Security settings
- Configure API keys
  * Service access
  * Authentication
  * Rate limiting
  * Usage tracking
- Set environment-specific variables
  * Development settings
  * Production configuration
  * Testing parameters
  * Debug options

3. Initialize the database:
```bash
python app.py
```

## Project Structure

```
├── api/                    # API route definitions
│   ├── user_routes.py     # User endpoints
│   │   ├── Question handling
│   │   ├── Response generation
│   │   └── Conversation management
│   ├── doctor_routes.py   # Doctor endpoints
│   │   ├── Response validation
│   │   ├── Quality assessment
│   │   └── Correction management
│   └── admin_routes.py    # Admin endpoints
│       ├── System management
│       ├── Knowledge base control
│       └── Performance monitoring
├── services/              # Core business logic
│   ├── user/             # User-related services
│   │   ├── retrieval.py  # Knowledge retrieval
│   │   │   ├── PCST algorithm
│   │   │   ├── Graph traversal
│   │   │   └── Context extraction
│   │   ├── response_generation.py # Response generation
│   │   │   ├── Text generation
│   │   │   ├── Context integration
│   │   │   └── Quality control
│   │   └── evaluation.py # Response evaluation
│   │       ├── Quality metrics
│   │       ├── Performance tracking
│   │       └── Improvement analysis
│   ├── admin/            # Admin services
│   │   ├── admin_services.py # Admin operations
│   │   │   ├── System management
│   │   │   ├── Configuration control
│   │   │   └── Performance monitoring
│   │   ├── bechmarking.py # Performance evaluation
│   │   │   ├── System metrics
│   │   │   ├── Quality assessment
│   │   │   └── Performance analysis
│   │   ├── buiding_augmenting_knowledge_base.py # KB management
│   │   │   ├── Data processing
│   │   │   ├── Graph construction
│   │   │   └── Knowledge integration
│   │   └── scrapping.py  # Data collection
│   │       ├── Web scraping
│   │       ├── Content extraction
│   │       └── Data cleaning
│   └── embedding_models.py # Embedding model implementations
│       ├── Model loading
│       ├── Text processing
│       └── Vector generation
├── models/               # Database models
│   ├── User management
│   ├── Conversation tracking
│   ├── Response storage
│   └── Validation records
├── knowledge_base/       # Knowledge storage and retrieval
│   ├── Graph data
│   ├── Vector embeddings
│   ├── Raw documents
│   └── Processed data
├── evaluation/          # System evaluation tools
│   ├── Quality metrics
│   ├── Performance analysis
│   └── Improvement tracking
├── config.py            # Configuration settings
│   ├── System parameters
│   ├── API configurations
│   └── Environment variables
├── db.py               # Database setup
│   ├── Connection management
│   ├── Schema definition
│   └── Query optimization
└── app.py              # Application entry point
    ├── Server initialization
    ├── Route registration
    └── Error handling
```

## Security Features

- User authentication and authorization
  * Secure login
  * Role-based access
  * Session management
  * Token validation
- Role-based access control
  * Permission management
  * Access restrictions
  * Activity logging
  * Security auditing
- Input validation
  * Data sanitization
  * Type checking
  * Format validation
  * Security checks
- Secure API endpoints
  * HTTPS enforcement
  * Request validation
  * Response sanitization
  * Error handling
- Protected knowledge base access
  * Access control
  * Data encryption
  * Audit logging
  * Version control
- API key management
  * Secure storage
  * Rotation policies
  * Usage tracking
  * Access control
- Environment variable protection
  * Secure storage
  * Access control
  * Encryption
  * Audit logging

## Performance Considerations

- Efficient vector search using Milvus
  * Index optimization
  * Query caching
  * Batch processing
  * Resource management
- Optimized database queries with proper indexing
  * Query optimization
  * Index management
  * Connection pooling
  * Cache utilization
- Conversation history management
  * Efficient storage
  * Quick retrieval
  * Context preservation
  * State management
- Response caching
  * Result storage
  * Cache invalidation
  * Memory management
  * Performance optimization
- Batch processing capabilities
  * Parallel processing
  * Resource optimization
  * Error handling
  * Progress tracking
- Streaming response generation
  * Real-time processing
  * Memory efficiency
  * Progress tracking
  * Error handling
- Efficient graph operations
  * Algorithm optimization
  * Memory management
  * Parallel processing
  * Cache utilization

## Future Improvements

### Advanced Features
- Real-time response streaming
  * Progress tracking
  * Partial results
  * Error handling
  * Resource management
- Enhanced validation workflows
  * Quality metrics
  * Expert review
  * Feedback integration
  * Performance tracking
- Advanced analytics dashboard
  * System metrics
  * Usage statistics
  * Performance data
  * Quality assessment
- Automated quality assessment
  * Metric calculation
  * Performance tracking
  * Improvement identification
  * Quality control
- Multi-language support
  * Translation
  * Localization
  * Cultural adaptation
  * Language detection
- Enhanced error handling
  * Comprehensive logging
  * Error tracking
  * Recovery mechanisms
  * User feedback
- Comprehensive logging system
  * Activity tracking
  * Performance monitoring
  * Error logging
  * Audit trails

### Technical Improvements
- Caching layer implementation
  * Result caching
  * Cache invalidation
  * Memory management
  * Performance optimization
- Rate limiting
  * Request throttling
  * Resource management
  * Usage tracking
  * Fair access
- Advanced error handling
  * Error recovery
  * User feedback
  * System stability
  * Debug information
- Performance optimization
  * Resource management
  * Query optimization
  * Cache utilization
  * Parallel processing
- Extended API documentation
  * Endpoint details
  * Usage examples
  * Error handling
  * Best practices
- Async/await implementation
  * Non-blocking operations
  * Resource efficiency
  * Scalability
  * Performance improvement
- Retry mechanisms for API calls
  * Error recovery
  * Resource management
  * Performance optimization
  * Reliability improvement
- Proxy support for web scraping
  * IP rotation
  * Rate limiting
  * Access management
  * Security enhancement
- Enhanced data validation
  * Input checking
  * Format validation
  * Security verification
  * Quality control
- Unit test implementation
  * Code coverage
  * Quality assurance
  * Regression testing
  * Performance testing
