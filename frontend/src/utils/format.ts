/**
 * Utility functions for formatting strings or dates.
 *
 * These helpers can be expanded as the UI becomes more sophisticated.
 */

/**
 * Truncate a string to a given length, adding an ellipsis if necessary.
 *
 * @param value - Input string
 * @param length - Maximum number of characters
 * @returns Truncated string
 */
export function truncate(value: string, length: number): string {
  return value.length > length ? `${value.slice(0, length)}…` : value;
}