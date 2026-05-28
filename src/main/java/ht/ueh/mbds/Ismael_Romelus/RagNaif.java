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
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class RagNaif {


    private static void configureLogger() {
        // Configure le logger sous-jacent (java.util.logging)
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE); // Ajuster niveau
        // Ajouter un handler pour la console pour faire afficher les logs
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
        packageLogger.setUseParentHandlers(false);
    }
    public static void main(String[] args) {

        configureLogger();
        String cle = System.getenv("GEMINI_KEY");

        // --- Modèles ---
        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(cle)
                .modelName("gemini-2.5-flash")
                .logRequests(true)
                .logResponses(true)
                .build();

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // Phase 1 du RAG  :  enregistrement des embeddings qui seront utilisés dans la phase 2.


        // 1. Chargement du document PDF
        Document document = ClassPathDocumentLoader.loadDocument("rag.pdf", new ApacheTikaDocumentParser());

        // 2. Découpage en segments (chunks : 300 tokens max, 30 de chevauchement)
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);

        // 3. Création des embeddings pour tous les segments
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        // 4. Stockage dans le magasin d'embeddings en mémoire
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);


        // PHASE 2 : utilisation des embeddings pour répondre aux questions.


        // ContentRetriever : top 2 résultats, score >= 0.5
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        //  Mémoire de conversation (10 messages)
        //  Création de l'assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .contentRetriever(contentRetriever)
                .build();

        // 8. Boucle de questions
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
