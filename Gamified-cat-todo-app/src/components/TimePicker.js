import React from 'react';

export function TimePicker({ date, setDate }) {
  const handleChange = (e) => {
    const newDate = new Date(date);
    const [hour, minute] = e.target.value.split(":").map(Number);
    newDate.setHours(hour, minute);
    setDate(newDate);
  };

  return (
    <input
      type="time"
      value={date.toISOString().substr(11, 5)} // Format HH:mm
      onChange={handleChange}
      className="border p-2 rounded-md"
    />
  );
}