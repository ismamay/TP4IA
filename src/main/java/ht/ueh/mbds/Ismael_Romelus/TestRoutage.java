package ht.ueh.mbds.Ismael_Romelus;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;


public class TestRoutage {

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
        packageLogger.setUseParentHandlers(false);
    }

    private static EmbeddingStore<TextSegment> ingest(String nomFichier, EmbeddingModel embeddingModel) {
        Document document = ClassPathDocumentLoader.loadDocument(nomFichier, new ApacheTikaDocumentParser());
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);
        return store;
    }

    public static void main(String[] args) {

        configureLogger();
        String cle = System.getenv("GEMINI_KEY");

        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(cle)
                .modelName("gemini-2.5-flash")
                .logRequests(true)
                .logResponses(true)
                .build();

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // Phase 1 - Ingestion des 2 documents
        EmbeddingStore<TextSegment> storeRag = ingest("rag.pdf", embeddingModel);
        EmbeddingStore<TextSegment> storeLangchain = ingest("langchain4j.pdf", embeddingModel);

        // Phase 2 - 2 ContentRetrievers
        ContentRetriever retrieverRag = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeRag)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        ContentRetriever retrieverLangchain = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeLangchain)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // QueryRouter - le LLM choisit le bon retriever
        Map<ContentRetriever, String> descriptions = new HashMap<>();
        descriptions.put(retrieverRag, "Support de cours sur le RAG (Retrieval-Augmented Generation) et le fine-tuning");
        descriptions.put(retrieverLangchain, "Support de cours sur LangChain4j, la librairie Java pour les LLMs");

        QueryRouter queryRouter = new LanguageModelQueryRouter(chatModel, descriptions);

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        // Boucle de questions
        Scanner scanner = new Scanner(System.in);
        System.out.println("Posez vos questions (tapez 'exit' pour quitter) :");
        while (true) {
            System.out.print("\nVous : ");
            String question = scanner.nextLine();
            if ("exit".equalsIgnoreCase(question)) break;
            String reponse = assistant.chat(question);
            System.out.println("Assistant : " + reponse);
        }
    }


}
