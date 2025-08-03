import Head from 'next/head';
import { useEffect, useState } from 'react';
import {
  downloadDataset,
  getDatasetStatus,
  getDatasetPrepareStatus,
  prepareDataset,
} from '../services/api';

/**
 * Dataset management page.
 *
 * Allows the user to trigger a dataset download and view its status.
 */
export default function DatasetPage() {
  const [status, setStatus] = useState('not started');
  const [prepareStatus, setPrepareStatus] = useState('not started');
  const [loading, setLoading] = useState(false);

  const refreshStatus = async () => {
    const current = await getDatasetStatus();
    setStatus(current);
    const prep = await getDatasetPrepareStatus();
    setPrepareStatus(prep);
  };

  useEffect(() => {
    refreshStatus();
  }, []);

  const handleDownload = async () => {
    setLoading(true);
    await downloadDataset();
    await refreshStatus();
    setLoading(false);
  };

  return (
    <>
      <Head>
        <title>Dataset Management</title>
      </Head>
      <main style={{ padding: '2rem', maxWidth: '800px', margin: '0 auto' }}>
        <h1>Dataset Management</h1>
        <p>Download status: {status}</p>
        <p>Prepare status: {prepareStatus}</p>
        <button onClick={handleDownload} disabled={loading}>
          {loading ? 'Downloading...' : 'Download Dataset'}
        </button>
        <button
          onClick={async () => {
            setLoading(true);
            await prepareDataset();
            await refreshStatus();
            setLoading(false);
          }}
          disabled={loading || status !== 'completed'}
          style={{ marginLeft: '1rem' }}
        >
          {loading ? 'Preparing...' : 'Prepare Dataset'}
        </button>
      </main>
    </>
  );
}