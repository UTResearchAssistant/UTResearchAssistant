import Head from 'next/head';
import { useState } from 'react';
import SearchBar from '../components/SearchBar';
import SummaryCard from '../components/SummaryCard';
import { searchDocuments, summariseText } from '../services/api';

/**
 * Home page of the research assistant.  Users can enter a query,
 * perform a search and view a summary of selected results.  This
 * component demonstrates how to use the API services defined in
 * ``src/services/api.ts``.
 */
export default function Home() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<string[]>([]);
  const [summary, setSummary] = useState<string>('');

  const handleSearch = async () => {
    const res = await searchDocuments(query);
    setResults(res);
  };

  const handleSummarise = async (text: string) => {
    const { summary } = await summariseText(text);
    setSummary(summary);
  };

  return (
    <>
      <Head>
        <title>AI Research Assistant</title>
        <meta name="description" content="AI‑powered research assistant" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>
      <main style={{ padding: '2rem', maxWidth: '800px', margin: '0 auto' }}>
        <h1>AI Research Assistant</h1>
        <SearchBar
          value={query}
          onChange={setQuery}
          onSearch={handleSearch}
        />
        {results.length > 0 && (
          <section>
            <h2>Search Results</h2>
            <ul>
              {results.map((item) => (
                <li key={item} style={{ marginBottom: '1rem' }}>
                  {item}{' '}
                  <button onClick={() => handleSummarise(item)}>Summarise</button>
                </li>
              ))}
            </ul>
          </section>
        )}
        {summary && <SummaryCard summary={summary} />}
      </main>
    </>
  );
}