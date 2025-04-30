import React from 'react';

export function Select({ value, onChange, children, className }) {
  return (
    <select
      value={value}
      onChange={onChange}  // This should update the value correctly in the parent component's state
      className={`px-4 py-2 border rounded-md ${className}`}
    >
      {children}
    </select>
  );
}

export function SelectTrigger({ children, className }) {
  return (
    <div className={`cursor-pointer ${className}`}>
      {children}
    </div>
  );
}

export function SelectContent({ children, className }) {
  return (
    <div className={`p-2 ${className}`}>
      {children}
    </div>
  );
}

export function SelectItem({ value, children }) {
  return <option value={value}>{children}</option>;
}