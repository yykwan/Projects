import React, { useState } from 'react';

export function Popover({ children, className }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className={`${className}`}>
      <div onClick={() => setIsOpen(!isOpen)} className="cursor-pointer">
        {children[0]} {/* Popover trigger */}
      </div>
      {isOpen && (
        <div className="absolute bg-white border shadow-md mt-2 p-2 rounded-md">
          {children[1]} {/* Popover content */}
        </div>
      )}
    </div>
  );
}

export function PopoverTrigger({ children }) {
  return <div>{children}</div>;
}

export function PopoverContent({ children }) {
  return <div>{children}</div>;
}