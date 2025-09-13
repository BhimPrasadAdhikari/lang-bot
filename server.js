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
let memory = []

/**
 * Transform user query into a standalone question
 * @param {string} question
 * @returns {Promise<string>} rewritten question
 */
async function transformQuery(question) {
  
    memory.push({ role: "user", content: question });
  

  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: memory.map(m => ({
      role: m.role,
      parts: [{text: m.content}]
    })),
    config: {
      systemInstruction: `You are a query rewriting expert. 
Based on the conversation history, rephrase the latest user question into a complete, 
standalone question that can be understood without chat history.

Output only the rewritten question.`,
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
    const crisisKeywords = ["suicide", "ആത്മഹത്യ", "आत्महत्या", "आत्महत्या करना", "kill myself"];
    if (crisisKeywords.some(k => question.toLowerCase().includes(k))) {
      return res.json({
        answer: `⚠️ I’m really concerned about what you just shared.  
If you are in immediate danger, please call your local emergency number right now.  

📞 India: Call 1800-599-0019 (Vandrevala Foundation) or 9152987821 (AASRA Helpline)  
📞 International: Visit https://findahelpline.com for crisis hotlines worldwide.  

You are not alone. Please reach out to someone you trust or a professional right away.`
      });
    }

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

    const response = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: [
        ...memory.map(m => ({
          role: m.role,
          parts: [{ text: m.content }]
        })),
        { role: "user", parts: [{ text: rewrittenQuestion }] }
      ],

      config: {
            systemInstruction: `You are a Agriculture Expert.
You will be given a context of relevant information and a user question.
Your task is to answer the user's question based ONLY on the provided context.

If the knowledge base context is missing or incomplete,  
still provide **general best practices** and politely ask a follow-up question  
to better guide the farmer.

IMPORTANT:  
- First, detect the language of the user’s question.  
- Reply strictly in the same language as the user (English → English, Malayalam → Malayalam, Hindi → Hindi, Nepali → Nepali, etc.).  
- Do NOT mix languages.  

Context:
${context}`,
      },
    });

    memory.push({role: "model", content:response.text})

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
