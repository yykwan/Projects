import React from 'react';

export function Card({ children, className }) {
  return (
    <div className={`p-4 bg-white shadow-lg rounded-lg ${className}`}>
      {children}
    </div>
  );
}

export function CardContent({ children }) {
  return <div className="p-2">{children}</div>;
}