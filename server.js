import * as dotenv from "dotenv";
dotenv.config();

import express from "express";
import cors from "cors";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { GoogleGenAI } from "@google/genai";

const app = express();
app.use(cors());
app.use(express.json());

// --- Initialize AI ---
const ai = new GoogleGenAI({});

/**
 * Transform user query into a standalone question
 * @param {string} question
 * @returns {Promise<string>} rewritten question
 */
async function transformQuery(question) {
  const history = [
    { role: "user", parts: [{ text: question }] }
  ];

  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: history,
    config: {
      systemInstruction: `You are a query rewriting expert. Based on the provided chat history, rephrase the "Follow Up user Question" into a complete, standalone question that can be understood without the chat history.
Only output the rewritten question and nothing else.`,
    },
  });

  return response.text;
}

/**
 * Chatbot endpoint: takes a question, retrieves context from Pinecone,
 * and generates a Gemini response.
 */
app.post("/chat", async (req, res) => {
  const question = req.body?.question?.trim();

  if (!question) {
    return res.status(400).json({ error: "❗ No question provided." });
  }

  try {
    // Step 1: Rewrite question
    const rewrittenQuestion = await transformQuery(question);

    // Step 2: Generate embedding vector
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: "text-embedding-004",
    });
    const queryVector = await embeddings.embedQuery(rewrittenQuestion);

    // Step 3: Query Pinecone
    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    const searchResults = await pineconeIndex.query({
      topK: 10,
      vector: queryVector,
      includeMetadata: true,
    });

    const context = searchResults.matches
      ?.map((match) => match.metadata?.text || "")
      .filter(Boolean)
      .join("\n\n---\n\n") || "No relevant context found.";

    // Step 4: Generate response from Gemini
    const localHistory = [
      { role: "user", parts: [{ text: rewrittenQuestion }] }
    ];

    const response = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: localHistory,
      config: {
            systemInstruction: `You are a Kerala Agriculture Expert.
You will be given a context of relevant information and a user question.
Your task is to answer the user's question based ONLY on the provided context.

If the knowledge base context is missing or incomplete,  
still provide **general best practices** and politely ask a follow-up question  
to better guide the farmer.

IMPORTANT:
- Detect the language of the question first.
- If the question is in English, reply ONLY in English.
- If the question is in Malayalam, reply ONLY in Malayalam.
- Do NOT mix languages.

Context:
${context}`,
      },
    });

    return res.json({ answer: response.text });
  } catch (error) {
    console.error("❌ Error in /chat endpoint:", error);
    return res.status(500).json({ error: "Something went wrong while processing your request." });
  }
});

// --- Start Server ---
const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
