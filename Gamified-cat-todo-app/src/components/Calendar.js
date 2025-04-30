import React from 'react';
import Calendar from 'react-calendar';
import 'react-calendar/dist/Calendar.css';

export function CustomCalendar({ selectedDate, onSelectDate }) {
  return (
    <Calendar
      onChange={onSelectDate}
      value={selectedDate}
    />
  );
}