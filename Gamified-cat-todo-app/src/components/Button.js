import React from 'react';

export function Button({ children, onClick, className }) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-700 ${className}`}
    >
      {children}
    </button>
  );
}