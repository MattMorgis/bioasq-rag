```mermaid
graph TD
    InputData[PubMed JSONL Corpus]

    subgraph ProcessingLayer["Processing Layer"]
        DataLoader[Data Loader]
        Pipeline[Pipeline Orchestrator]
    end

    subgraph CorePipeline["Core Pipeline (Per Document)"]
        direction LR

        AbstractChunkerInterface[AbstractChunker Interface]
        EmbedderInterface[Embedder Interface]

        PubMedAbstract[PubMedAbstract] --> AbstractChunkerInterface
        AbstractChunkerInterface --> |"yields"| PubMedChunk[PubMedChunk]
        PubMedChunk --> EmbedderInterface
        EmbedderInterface --> |"yields"| PubMedEmbeddedChunk[PubMedEmbeddedChunk]
    end

    subgraph ChunkerImplementations["Chunker Implementations"]
        direction TB
        AbstractChunkerInterface --> WordChunker[WordChunker]
    end

    subgraph EmbedderImplementations["Embedder Implementations"]
        direction TB
        EmbedderInterface --> SentenceTransformerEmbedder[SentenceTransformerEmbedder]
    end

    subgraph OutputOptions["Output Options"]
        direction TB
        VectorDB[Vector Database]
        FileSystem[File System Storage]
    end

    InputData --> DataLoader
    DataLoader --> Pipeline
    Pipeline --> PubMedAbstract

    PubMedEmbeddedChunk --> OutputOptions

    %% Pipeline Configuration
    Pipeline -.-> |"configures"| AbstractChunkerInterface
    Pipeline -.-> |"configures"| EmbedderInterface

    %% Pipeline Steps
    subgraph PipelineSteps["Pipeline Steps"]
        PipelineChunk[CHUNK]
        PipelineEmbed[EMBED]
    end

    Pipeline -.-> PipelineSteps

    classDef interface fill:#f9f9f9,stroke:#666,stroke-width:2px
    classDef implementation fill:#e3f2fd,stroke:#2196f3
    classDef storage fill:#e8f5e9,stroke:#4caf50
    classDef process fill:#fff8e1,stroke:#ffc107
    classDef data fill:#f3e5f5,stroke:#9c27b0
    classDef step fill:#ffebee,stroke:#f44336

    class AbstractChunkerInterface,EmbedderInterface interface
    class WordChunker,SentenceTransformerEmbedder implementation
    class VectorDB,FileSystem storage
    class DataLoader,Pipeline process
    class PubMedAbstract,PubMedChunk,PubMedEmbeddedChunk data
    class PipelineChunk,PipelineEmbed step
```
