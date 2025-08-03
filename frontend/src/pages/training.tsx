import Head from 'next/head';
import { useEffect, useState } from 'react';
import { startTraining, getTrainingStatus } from '../services/api';

/**
 * Training page.
 *
 * Provides controls to start a model training job and displays the current
 * status.  The actual training logic is a stub on the backend.
 */
export default function TrainingPage() {
  const [status, setStatus] = useState('not started');
  const [running, setRunning] = useState(false);

  const refreshStatus = async () => {
    const s = await getTrainingStatus();
    setStatus(s);
  };

  useEffect(() => {
    refreshStatus();
  }, []);

  const handleStart = async () => {
    setRunning(true);
    await startTraining();
    await refreshStatus();
    setRunning(false);
  };

  return (
    <>
      <Head>
        <title>Model Training</title>
      </Head>
      <main style={{ padding: '2rem', maxWidth: '800px', margin: '0 auto' }}>
        <h1>Model Training</h1>
        <p>Training status: {status}</p>
        <button onClick={handleStart} disabled={running}>
          {running ? 'Training...' : 'Start Training'}
        </button>
      </main>
    </>
  );
}