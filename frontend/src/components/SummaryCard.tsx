interface SummaryCardProps {
  summary: string;
}

/**
 * Displays a summary in a card‑like container.  In the future this
 * component could render references as footnotes, highlight entities,
 * or provide actions like saving to a library.  For now it renders
 * plain text.
 */
export default function SummaryCard({ summary }: SummaryCardProps) {
  return (
    <div
      style={{
        border: '1px solid #ccc',
        borderRadius: '4px',
        padding: '1rem',
        marginTop: '1rem',
      }}
    >
      <h3>Summary</h3>
      <p>{summary}</p>
    </div>
  );
}