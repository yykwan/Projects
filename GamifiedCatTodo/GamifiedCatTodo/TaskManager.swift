//
//  TaskManager.swift
//  GamifiedCatTodo
//
//  Created by Kwan Yi Yan on 2/5/2025.
//
import Foundation
import UserNotifications

class TaskManager: ObservableObject {
    @Published var tasks: [Task] = [] {
        didSet { saveTasks() }
    }

    @Published var coins = 0
    @Published var hearts = 9
    @Published var happiness = 0
    @Published var fishInventory = 0

    private let tasksKey = "tasks"
    private let coinsKey = "coins"
    private let heartsKey = "hearts"
    private let happinessKey = "happiness"
    private let fishKey = "fishInventory"

    init() {
        loadData()
        applyOverduePenalties()
        NotificationManager.shared.requestAuthorization()
        rescheduleNotifications()
    }

    // MARK: - Task Management
    func addTask(title: String, category: String, priority: TaskPriority, dueDate: Date?, wantsNotification: Bool = false) {
        let newTask = Task(
            title: title,
            category: category,
            priority: priority,
            dueDate: dueDate,
            wantsNotification: wantsNotification,
            notificationDate: dueDate?.addingTimeInterval(-3600) // 1 hour before
        )

        if wantsNotification {
            if let notifyDate = newTask.notificationDate, notifyDate > Date() {
                NotificationManager.shared.scheduleNotification(for: newTask)
            }
        }

        tasks.append(newTask)
    }

    func updateTaskNotification(
        _ task: Task,
        wantsNotification: Bool,
        reminderTimeBeforeDue: TimeInterval? = 3600
    ) {
        guard let index = tasks.firstIndex(where: { $0.id == task.id }) else { return }

        tasks[index].wantsNotification = wantsNotification
        tasks[index].notificationDate = task.dueDate?.addingTimeInterval(-(reminderTimeBeforeDue ?? 3600))

        NotificationManager.shared.cancelNotification(for: task.id)

        if wantsNotification, let notifyDate = tasks[index].notificationDate, notifyDate > Date() {
            NotificationManager.shared.scheduleNotification(for: tasks[index])
        }

        saveData()
    }

    func toggleCompletion(for task: Task) {
        guard let index = tasks.firstIndex(where: { $0.id == task.id }) else { return }

        tasks[index].isCompleted.toggle()

        if tasks[index].isCompleted {
            coins += 10
            NotificationManager.shared.cancelNotification(for: task.id)
        } else {
            coins = max(0, coins - 10)
            if tasks[index].wantsNotification {
                NotificationManager.shared.scheduleNotification(for: tasks[index])
            }
        }

        saveData()
    }

    func deleteTask(at indexSet: IndexSet) {
        indexSet.forEach { index in
            NotificationManager.shared.cancelNotification(for: tasks[index].id)
        }
        tasks.remove(atOffsets: indexSet)
        saveData()
    }

    // MARK: - Notification Handling
    func rescheduleNotifications() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
            NotificationManager.shared.rescheduleAllNotifications(for: self.tasks)
        }
    }

    // MARK: - Game Economy
    func buyFish() {
        if coins >= 10 {
            coins -= 10
            fishInventory += 1
            saveData()
        }
    }

    func buyHeart() {
        if coins >= 50 && hearts < 9 {
            coins -= 50
            hearts += 1
            saveData()
        }
    }

    func feedFish() {
        if fishInventory > 0 {
            fishInventory -= 1
            happiness += 10
            saveData()
        }
    }

    // MARK: - Penalty System
    private func applyOverduePenalties() {
        let now = Date()
        var lostHearts = 0

        for task in tasks where !task.isCompleted {
            if let due = task.dueDate, due < now {
                lostHearts += 1
                NotificationManager.shared.cancelNotification(for: task.id)
            }
        }

        hearts = max(0, hearts - lostHearts)
        if lostHearts > 0 {
            saveData()
        }
    }

    // MARK: - Persistence
    private func saveTasks() {
        if let encoded = try? JSONEncoder().encode(tasks) {
            UserDefaults.standard.set(encoded, forKey: tasksKey)
        }
    }

    func saveData() {
        saveTasks()
        UserDefaults.standard.set(coins, forKey: coinsKey)
        UserDefaults.standard.set(hearts, forKey: heartsKey)
        UserDefaults.standard.set(happiness, forKey: happinessKey)
        UserDefaults.standard.set(fishInventory, forKey: fishKey)
    }

    private func loadData() {
        if let data = UserDefaults.standard.data(forKey: tasksKey),
           let decoded = try? JSONDecoder().decode([Task].self, from: data) {
            tasks = decoded
        }
        coins = UserDefaults.standard.integer(forKey: coinsKey)
        hearts = UserDefaults.standard.integer(forKey: heartsKey)
        happiness = UserDefaults.standard.integer(forKey: happinessKey)
        fishInventory = UserDefaults.standard.integer(forKey: fishKey)
    }
}
