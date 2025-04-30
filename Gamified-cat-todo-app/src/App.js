import React, { useState, useEffect } from "react";
import { Button } from "./components/Button";
import { Card, CardContent } from "./components/Card";
import { Input } from "./components/Input";
import {
  Select,
  SelectTrigger,
  SelectContent,
  SelectItem,
} from "./components/Select";
import { CustomCalendar } from "./components/Calendar";
import { Popover, PopoverTrigger, PopoverContent } from "./components/Popover";
import { TimePicker } from "./components/TimePicker";
import { format, isBefore } from 'date-fns';
import { unstable_batchedUpdates } from 'react-dom';



const initialFoods = ["Fish", "Milk", "Tuna"];
const MAX_HEARTS = 9;

export default function GamifiedTodoApp() {
  const [tasks, setTasks] = useState([]);
  const [newTask, setNewTask] = useState("");
  const [category, setCategory] = useState("");
  const [customCategory, setCustomCategory] = useState("");
  const [categories, setCategories] = useState(["Work", "Study", "Personal"]);
  const [priority, setPriority] = useState("Low");
  const [deadlineDate, setDeadlineDate] = useState(new Date());
  const [deadlineTime, setDeadlineTime] = useState(new Date());
  const [coins, setCoins] = useState(0);
  const [petInventory, setPetInventory] = useState([]);
  const [selectedFood, setSelectedFood] = useState(initialFoods[0]);
  const [petHappiness, setPetHappiness] = useState(0);
  const [hearts, setHearts] = useState(MAX_HEARTS);
  const [error, setError] = useState("");
  const [notifications, setNotifications] = useState([]);

  useEffect(() => {
    const now = new Date();
    const overdueTasks = tasks.filter(
      task => !task.completed && 
             !task.penaltyApplied && 
             isBefore(task.deadline, now)
    );
  
    if (overdueTasks.length > 0) {
      // Process all overdue tasks at once
      const penaltyCount = overdueTasks.length;
      
      unstable_batchedUpdates(() => {
        // Reduce hearts by number of overdue tasks
        setHearts(h => Math.max(0, h - penaltyCount));
        
        // Mark all as penalized
        setTasks(prevTasks => 
          prevTasks.map(t => 
            overdueTasks.some(ot => ot.id === t.id)
              ? { ...t, penaltyApplied: true }
              : t
          )
        );
        
        // Add notifications
        setNotifications(prev => [
          ...prev,
          ...overdueTasks.map(task => ({
            id: Date.now() + task.id,
            message: `"${task.text}" was overdue! -1 heart`,
            type: 'penalty'
          }))
        ]);
      });
    }
  }, [tasks]);

  useEffect(() => {
    console.log('Coins updated to:', coins);
  }, [coins]);
  useEffect(() => {
    console.log("Current hearts:", hearts);
  }, [hearts]);

  const addTask = () => {
    setError("");

    // Validate task description
    if (!newTask.trim()) {
      setError("Please enter a task description");
      return;
    }

    // Validate category
    if (!category) {
      setError("Please select a category");
      return;
    }

    // If custom category is selected but empty
    if (category === "custom" && !customCategory.trim()) {
      setError("Please enter a custom category");
      return;
    }

    // Determine the final category
    const selectedCategory = category === "custom" ? customCategory.trim() : category;

    // Add new custom category to list if needed
    if (category === "custom" && !categories.includes(selectedCategory)) {
      setCategories([...categories, selectedCategory]);
    }

    // Create deadline
    const deadline = new Date(
      deadlineDate.getFullYear(),
      deadlineDate.getMonth(),
      deadlineDate.getDate(),
      deadlineTime.getHours(),
      deadlineTime.getMinutes()
    );

    const newTaskObj = {
      id: Date.now(),
      text: newTask.trim(),
      category: selectedCategory,
      priority,
      deadline,
      completed: false,
      overdue: false, // Add this new field
      penaltyApplied: false // Add this
    };


    // Create new task
    setTasks([...tasks, newTaskObj]);

    // Reset form
    setNewTask("");
    setCategory("");
    setCustomCategory("");
    setPriority("Low");
    setDeadlineDate(new Date());
    setDeadlineTime(new Date());
  };

  const sortedTasks = [...tasks].sort((a, b) => {
    const priorityOrder = ["Low", "Medium", "High"];
    return priorityOrder.indexOf(a.priority) - priorityOrder.indexOf(b.priority);
  });

  function Notification({ message, type, onDismiss }) {
    const bgColor = type === 'overdue' ? 'bg-red-100 border-red-400 text-red-700' :
      type === 'completed' ? 'bg-green-100 border-green-400 text-green-700' :
        'bg-blue-100 border-blue-400 text-blue-700';

    return (
      <div className={`${bgColor} border px-4 py-3 rounded relative mb-2`}>
        <span className="block sm:inline">{message}</span>
        <button
          onClick={onDismiss}
          className="absolute top-0 bottom-0 right-0 px-4 py-3"
        >
          ×
        </button>
      </div>
    );
  }


  const handleCheckboxChange = (taskId) => {
    unstable_batchedUpdates(() => {
      setTasks(prevTasks =>
        prevTasks.map(task =>
          task.id === taskId
            ? { ...task, completed: !task.completed }
            : task
        )
      );

      setCoins(prev => {
        const task = tasks.find(t => t.id === taskId);
        return task?.completed ? Math.max(0, prev - 10) : prev + 10;
      });
    });
  };

  const buyFood = () => {
    if (coins >= 10) {
      setCoins(coins - 10);
      setPetInventory([...petInventory, selectedFood]);
    }
  };

  const feedCat = (food) => {
    const index = petInventory.indexOf(food);
    if (index !== -1) {
      const newInventory = [...petInventory];
      newInventory.splice(index, 1);
      setPetInventory(newInventory);
      setPetHappiness(petHappiness + 10);
    }
  };

  const reviveHeart = () => {
    if (coins >= 50 && hearts < MAX_HEARTS) {
      setCoins(coins - 50);
      setHearts(hearts + 1);
    }
  };

  return (
    <div className="p-4 h-screen flex gap-4">
      <div className="w-1/4 space-y-6">
        <div>
          <h2 className="text-xl font-semibold mb-2">Shop</h2>
          <Select onValueChange={setSelectedFood} defaultValue={selectedFood}>
            <SelectTrigger>{selectedFood}</SelectTrigger>
            <SelectContent>
              {initialFoods.map((food, i) => (
                <SelectItem key={i} value={food}>{food}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button className="mt-2" onClick={buyFood}>Buy Food (10 coins)</Button>
          <Button
            onClick={reviveHeart}
            disabled={hearts >= MAX_HEARTS || coins < 50}
            className="mt-2 w-full"
          >
            Revive Heart (50 coins)
          </Button>
        </div>
        <div>
          <h2 className="text-xl font-semibold mb-2">Inventory</h2>
          <div className="flex flex-wrap gap-2">
            {petInventory.map((food, index) => (
              <Button key={index} variant="outline" onClick={() => feedCat(food)}>
                {food}
              </Button>
            ))}
          </div>
        </div>
        <div>
          <h2 className="text-xl font-semibold mb-2">Settings</h2>
          <Card><CardContent className="p-2">Coming Soon...</CardContent></Card>
        </div>
      </div>

      <div className="w-2/5 flex flex-col items-center justify-center relative">
        <div className="absolute top-0 mt-2 flex flex-col items-center">
          <div className="flex space-x-1 mb-2">
            {Array.from({ length: hearts }).map((_, i) => (  // Only render 'hearts' number of emojis
              <span key={i} className="text-2xl text-red-500">
                ❤️
              </span>
            ))}
          </div>
          <div className="text-base font-medium">
          Hearts: {hearts}/{MAX_HEARTS} | Pet Happiness: {petHappiness} | Coins: {coins}
          </div>
        </div>
        <img
          src="https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif"
          alt="Cute Cat"
          className="w-48 h-48 rounded-xl shadow mb-4 mt-16"
        />
      </div>
      <div className="absolute top-20 right-4 w-64">
        {notifications.map((notification) => (
          <Notification
            key={notification.id}
            message={notification.message}
            type={notification.type}
            onDismiss={() =>
              setNotifications(prev =>
                prev.filter(n => n.id !== notification.id)
              )
            }
          />
        ))}
      </div>
      <div className="w-1/3">
        <h2 className="text-2xl font-bold mb-4">To-Do List</h2>

        <Input
          placeholder="New Task"
          value={newTask}
          onChange={(e) => setNewTask(e.target.value)}
          className="mb-2"
        />

        <div className="flex gap-2 mb-2">
          {/* Category Select */}
          <div className="flex-1">
            <Select
              value={category}
              onChange={(e) => {
                setCategory(e.target.value);
                if (e.target.value !== "custom") setCustomCategory("");
              }}
              className="w-full"
            >
              <option value="">Select Category</option>
              {categories.map((cat) => (
                <SelectItem key={cat} value={cat}>{cat}</SelectItem>
              ))}
              <SelectItem value="custom">Custom Category</SelectItem>
            </Select>

            {category === "custom" && (
              <Input
                placeholder="Enter custom category"
                value={customCategory}
                onChange={(e) => setCustomCategory(e.target.value)}
                className="mt-2"
              />
            )}
          </div>

          {/* Priority Select */}
          <Select
            value={priority}
            onChange={(e) => setPriority(e.target.value)}
            className="w-32"
          >
            {["Low", "Medium", "High"].map((level) => (
              <SelectItem key={level} value={level}>{level}</SelectItem>
            ))}
          </Select>
        </div>

        <div className="flex items-center gap-2 mb-2">
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="outline">
                {format(deadlineDate, "PPP")}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0 z-50" align="start">
              <CustomCalendar
                selectedDate={deadlineDate}
                onSelectDate={(date) => {
                  if (date) {
                    const updated = new Date(date);
                    updated.setHours(deadlineTime.getHours());
                    updated.setMinutes(deadlineTime.getMinutes());
                    setDeadlineDate(updated);
                    setDeadlineTime(updated);
                  }
                }}
              />
            </PopoverContent>
          </Popover>

          <input
            type="time"
            value={format(deadlineTime, "HH:mm")}
            onChange={(e) => {
              const [hour, minute] = e.target.value.split(":").map(Number);
              const updated = new Date(deadlineDate);
              updated.setHours(hour);
              updated.setMinutes(minute);
              setDeadlineTime(updated);
              setDeadlineDate(updated);
            }}
            className="border p-2 rounded-md"
          />
        </div>

        {error && (
          <div className="mb-2 text-red-500 text-sm">{error}</div>
        )}

        <Button onClick={addTask} className="mb-4">
          Add
        </Button>

        <div className="mt-6 space-y-6 max-h-[400px] overflow-y-auto pr-2">
          <h2 className="text-3xl font-bold">Task List</h2>

          {tasks.length === 0 ? (
            <p className="text-gray-500">No tasks yet.</p>
          ) : (
            <ul className="space-y-6">
              {tasks.map((task) => {
                const deadline = new Date(task.deadline);
                const formattedTime = format(deadline, "h:mm a"); // Format time as "2:30 PM"
                const formattedDate = deadline.toLocaleDateString(); // Format date as "MM/DD/YYYY"

                return (
                  <li key={task.id} className="flex items-start gap-4 p-2 border rounded-lg">
                    <input
                      type="checkbox"
                      checked={task.completed}
                      onChange={() => handleCheckboxChange(task.id)}
                      className="mt-1 h-5 w-5 accent-blue-500"
                    />
                    <div className={`flex-1 ${task.completed ? "opacity-70" : ""}`}>
                      <div className={`text-xl font-medium ${task.completed ? "line-through text-gray-500" : ""}`}>
                        {task.text}
                      </div>
                      <div className={`mt-1 text-sm space-x-4 ${task.completed ? "text-gray-400" : "text-gray-600"}`}>
                        <span>📅 {formattedDate} at {formattedTime}</span>
                        <span>⭐ {task.priority}</span>
                        <span>🏷️ {task.category}</span>
                      </div>
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}