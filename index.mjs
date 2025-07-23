import dotenv from "dotenv";
dotenv.config();

import axios from "axios";
import readline from "readline";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

// 1. Figma 문서 텍스트 추출
async function fetchFigmaDocumentText() {
  const fileKey = process.env.FIGMA_FILE_KEY;
  const token = process.env.FIGMA_TOKEN;

  const response = await axios.get(
    `https://api.figma.com/v1/files/${fileKey}`,
    {
      headers: {
        "X-Figma-Token": token,
      },
    }
  );

  const traverse = (node, texts = []) => {
    if (node.name) texts.push(node.name);
    if (node.characters) texts.push(node.characters);
    if (node.children) {
      for (const child of node.children) {
        traverse(child, texts);
      }
    }
    return texts;
  };

  const document = response.data.document;
  const textContents = traverse(document).filter(Boolean).join("\n");
  return textContents;
}

// 2. 피그마 문서 읽기
const rawText = await fetchFigmaDocumentText();

// 3. 문서 쪼개기
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 100,
});
const docs = await splitter.createDocuments([rawText]);

// 4. HuggingFace 임베딩
const embeddings = new HuggingFaceInferenceEmbeddings({
  apiKey: process.env.HUGGINGFACE_API_KEY,
  model: "sentence-transformers/all-MiniLM-L6-v2",
});

// 5. 메모리 벡터스토어 구성
const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);

// 6. OpenRouter 기반 LLM 정의
const openRouterModel = {
  call: async (prompt) => {
    const res = await axios.post(
      "https://openrouter.ai/api/v1/chat/completions",
      {
        model: "mistralai/mistral-7b-instruct", // 무료 모델
        messages: [{ role: "user", content: prompt }],
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
          "Content-Type": "application/json",
        },
      }
    );
    return res.data.choices[0].message.content;
  },
};

// 7. CLI 인터페이스
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

console.log("📐 피그마 문서 기반 Q&A 봇입니다. 질문을 입력하세요:");
let chatHistory = [];

rl.on("line", async (line) => {
  const question = line.trim();
  const relevantDocs = await vectorStore.similaritySearch(question, 4);
  const context = relevantDocs.map((doc) => doc.pageContent).join("\n");

  const prompt = `
다음은 피그마 문서 내용입니다:

${context}

위 문서를 참고해서 다음 질문에 친절하고 간결하게 답해주세요:

질문: ${question}
`;

  const answer = await openRouterModel.call(prompt);
  console.log(`\n🤖 ${answer}\n`);
  chatHistory.push([question, answer]);
});
