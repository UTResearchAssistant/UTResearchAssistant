import { ChangeEvent } from 'react';

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  onSearch: () => void;
}

/**
 * A simple search bar component.  It accepts the current value,
 * change handler and a callback to perform the search.  Styling is kept
 * minimal; consumers can wrap it in a layout component to fit their
 * needs.
 */
export default function SearchBar({ value, onChange, onSearch }: SearchBarProps) {
  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    onChange(e.target.value);
  };

  return (
    <div style={{ marginBottom: '1rem' }}>
      <input
        type="text"
        value={value}
        onChange={handleInputChange}
        placeholder="Enter search term"
        style={{ padding: '0.5rem', width: '70%' }}
      />
      <button onClick={onSearch} style={{ marginLeft: '0.5rem' }}>
        Search
      </button>
    </div>
  );
}