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
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test5 {

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
        packageLogger.setUseParentHandlers(false);
    }

    public static void main(String[] args) {

        configureLogger();
        String cle = System.getenv("GEMINI_KEY");
        String tavily = System.getenv("TAVILY_KEY");

        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(cle)
                .modelName("gemini-2.5-flash")
                .logRequests(true)
                .logResponses(true)
                .build();

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // Phase 1 - Ingestion du document PDF
        Document document = ClassPathDocumentLoader.loadDocument("rag.pdf", new ApacheTikaDocumentParser());
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // Phase 2 - ContentRetriever PDF
        ContentRetriever pdfRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // ContentRetriever Web (Tavily)
        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavily)
                .build();

        ContentRetriever webRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                .build();


//        QueryRouter queryRouter = new DefaultQueryRouter(pdfRetriever, webRetriever);
        Map<ContentRetriever, String> descriptions = new HashMap<>();
        descriptions.put(pdfRetriever, "Support de cours sur le RAG (Retrieval-Augmented Generation) et le fine-tuning");
        descriptions.put(webRetriever, "Recherche web pour les événements récents et informations générales non liées à l'IA");
        QueryRouter queryRouter = new LanguageModelQueryRouter(chatModel, descriptions);
        
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        Scanner scanner = new Scanner(System.in);
        System.out.println("Posez vos questions (tapez 'exit' pour quitter) :");
        while (true) {
            System.out.print("\nVous : ");
            String question = scanner.nextLine();
            if ("exit".equalsIgnoreCase(question)) break;
            System.out.println("Assistant : " + assistant.chat(question));
        }
    }

}
