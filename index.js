import * as dotenv from 'dotenv';
dotenv.config();

import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';

// 📌 List all PDFs here
const PDF_FILES = [
  "./pop2016.pdf",
  "./ab.pdf",
  "./ac.pdf",
  "./ad.pdf",
  "./ae.pdf",
  "./af.pdf",
  "./ag.pdf",
  "./ah.pdf",
    "./ai.pdf",
      "./aj.pdf",
        "./ak.pdf",
          "./al.pdf",
            "./am.pdf",
              "./an.pdf",
                "./ao.pdf",
                  "./ap.pdf",
                    "./aq.pdf",
                      "./ar.pdf",
                        "./as.pdf",
                          "./at.pdf",
                            "./au.pdf",
                              "./av.pdf",
                                "./aw.pdf",
                                  "./ax.pdf",
                                    "./ay.pdf",

  // 👆 keep adding until all 25 PDFs are listed
];

async function indexDocuments() {
  // init Pinecone + Embedding once
  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: "text-embedding-004",
  });

  const pinecone = new Pinecone();
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
  console.log("✅ Pinecone configured");

  // loop through all PDFs
  for (const filePath of PDF_FILES) {
    console.log(`\n📂 Processing ${filePath} ...`);

    const pdfLoader = new PDFLoader(filePath);
    const rawDocs = await pdfLoader.load();

    // add metadata so you know which PDF each chunk came from
    rawDocs.forEach(doc => {
      doc.metadata = { source: filePath };
    });

    // chunking
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
    console.log(`✅ Chunking completed for ${filePath}`);

    // store in Pinecone
    await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
      pineconeIndex,
      maxConcurrency: 5,
    });

    console.log(`🎉 Stored successfully: ${filePath}`);
  }

  console.log("\n🚀 All PDFs processed and stored in Pinecone!");
}

indexDocuments();
