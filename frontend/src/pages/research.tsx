import Head from 'next/head';
import { useState } from 'react';
import { performResearch } from '../services/api';

/**
 * Deep research page.
 *
 * Provides a form for the user to enter a research question and displays
 * the consolidated answer produced by the deep research service.
 */
export default function ResearchPage() {
  const [query, setQuery] = useState('');
  const [result, setResult] = useState<null | { answer: string; sources: string[] }>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true);
    const res = await performResearch(query);
    setResult(res);
    setLoading(false);
  };

  return (
    <>
      <Head>
        <title>Deep Research</title>
      </Head>
      <main style={{ padding: '2rem', maxWidth: '800px', margin: '0 auto' }}>
        <h1>Deep Research</h1>
        <div style={{ marginBottom: '1rem' }}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter a research question"
            style={{ width: '70%', padding: '0.5rem' }}
          />
          <button onClick={handleSubmit} disabled={loading} style={{ marginLeft: '0.5rem' }}>
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>
        {result && (
          <section>
            <h2>Answer</h2>
            <p>{result.answer}</p>
            {result.sources && result.sources.length > 0 && (
              <div>
                <h3>Sources</h3>
                <ul>
                  {result.sources.map((src, idx) => (
                    <li key={idx}>{src}</li>
                  ))}
                </ul>
              </div>
            )}
          </section>
        )}
      </main>
    </>
  );
}