import axios from 'axios';

/**
 * API base URL.  During development this points at the FastAPI backend
 * running on localhost.  In production the URL should be set via
 * environment variable or Next.js runtime configuration.
 */
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

/**
 * Perform a search for documents.
 *
 * @param query - The search term
 * @returns A promise resolving to a list of result strings
 */
export async function searchDocuments(query: string): Promise<string[]> {
  const response = await axios.get(`${API_BASE_URL}/api/v1/search`, { params: { q: query } });
  return response.data.results;
}

interface SummariseResult {
  summary: string;
  references: string[];
}

/**
 * Request a summary of the given text from the backend.
 *
 * @param text - The text or document to summarise
 * @returns A promise resolving to a summary and its references
 */
export async function summariseText(text: string): Promise<SummariseResult> {
  const response = await axios.post(`${API_BASE_URL}/api/v1/summarise`, { text });
  return response.data;
}

// Dataset management

export async function downloadDataset(): Promise<string> {
  const response = await axios.post(`${API_BASE_URL}/api/v1/dataset/download`);
  return response.data.message;
}

export async function getDatasetStatus(): Promise<string> {
  const response = await axios.get(`${API_BASE_URL}/api/v1/dataset/status`);
  return response.data.status;
}

export async function getDatasetPrepareStatus(): Promise<string> {
  const response = await axios.get(`${API_BASE_URL}/api/v1/dataset/status`);
  return response.data.prepare_status;
}

export async function prepareDataset(): Promise<string> {
  const response = await axios.post(`${API_BASE_URL}/api/v1/dataset/prepare`);
  return response.data.message;
}

// Model training

export async function startTraining(): Promise<string> {
  const response = await axios.post(`${API_BASE_URL}/api/v1/train/start`);
  return response.data.message;
}

export async function getTrainingStatus(): Promise<string> {
  const response = await axios.get(`${API_BASE_URL}/api/v1/train/status`);
  return response.data.status;
}

// Deep research

export interface ResearchResult {
  answer: string;
  sources: string[];
}

export async function performResearch(query: string): Promise<ResearchResult> {
  const response = await axios.post(`${API_BASE_URL}/api/v1/research`, { query });
  return response.data;
}