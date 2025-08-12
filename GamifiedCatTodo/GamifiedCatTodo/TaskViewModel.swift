//
//  TaskViewModel.swift
//  GamifiedCatTodo
//
//  Created by Kwan Yi Yan on 2/5/2025.
//
import Foundation
import Combine
import UIKit
import SwiftUI
import UserNotifications

class TaskViewModel: ObservableObject {
    // MARK: - Main Game Data
    @Published var tasks: [Task] = [] {
        didSet { saveTasks() }
    }

    @Published var coins: Int = 0 {
        didSet { checkHeartCondition(); saveData() }
    }

    @Published var hearts: Int {
        didSet {
            UserDefaults.standard.set(hearts, forKey: "hearts")
            saveData()
        }
    }

    @Published var happiness: Int = 0 {
        didSet { saveData() }
    }

    @Published var fishInventory: Int = 0 {
        didSet { saveData() }
    }

    // MARK: - Achievement Data
    @Published var achievements: [Achievement] = []
    @Published var highestHeartsMaintained: Int = 0
    @Published var daysSurvived: Int = 0
    @Published var lastResetDate: Date = Date()

    private var cancellables = Set<AnyCancellable>()
    private let tasksKey = "tasks"
    private let coinsKey = "coins"
    private let heartsKey = "hearts"
    private let happinessKey = "happiness"
    private let fishKey = "fishInventory"
    private let lastDeductionTimestampKey = "lastDeductionTimestamp"

    init() {
        let savedHearts = UserDefaults.standard.object(forKey: heartsKey) as? Int ?? 9
            let savedCoins = UserDefaults.standard.integer(forKey: coinsKey)
            let savedHappiness = UserDefaults.standard.integer(forKey: happinessKey)
            let savedFish = UserDefaults.standard.integer(forKey: fishKey)

            // Initialize the @Published properties without triggering didSet
            self._hearts = Published(initialValue: savedHearts)
            self._coins = Published(initialValue: savedCoins)
            self._happiness = Published(initialValue: savedHappiness)
            self._fishInventory = Published(initialValue: savedFish)
        loadData()
        startContinuousOverdueCheck()
        checkForNewAchievements()
    }

    // MARK: - Task Management

    func addTask(_ task: Task) {
        tasks.append(task)
        scheduleNotification(for: task)
    }

    func updateTask(_ task: Task) {
        if let index = tasks.firstIndex(where: { $0.id == task.id }) {
            tasks[index] = task
            scheduleNotification(for: task)
        }
    }

    func toggleCompletion(for task: Task) {
        guard let index = tasks.firstIndex(where: { $0.id == task.id }) else { return }

        var updatedTask = tasks[index]
        let wasCompleted = updatedTask.isCompleted
        updatedTask.isCompleted.toggle()

        if updatedTask.isCompleted && !wasCompleted {
            coins += 10
            updatedTask.hasPenaltyApplied = false
        } else if !updatedTask.isCompleted && wasCompleted {
            coins = max(0, coins - 10)
        }

        // Replace the task to trigger save
        tasks[index] = updatedTask
    }

    func togglePinning(for task: Task) {
        if let index = tasks.firstIndex(where: { $0.id == task.id }) {
            tasks[index].isPinned.toggle()
            saveTasks()
            objectWillChange.send()
        }
    }

    func deleteTask(_ task: Task) {
        UNUserNotificationCenter.current().removePendingNotificationRequests(withIdentifiers: [task.id.uuidString])
        tasks.removeAll { $0.id == task.id }
    }

    func updateTaskPriority(for task: Task) {
        if let index = tasks.firstIndex(where: { $0.id == task.id }) {
            let newPriority: TaskPriority
            switch task.priority {
            case .low:
                newPriority = .medium
            case .medium:
                newPriority = .high
            case .high:
                newPriority = .low
            }
            tasks[index].priority = newPriority
            scheduleNotification(for: tasks[index])
        }
    }

    // MARK: - Notification Management

    func scheduleNotification(for task: Task) {
        guard task.shouldScheduleNotification, let date = task.notificationDate else { return }

        UNUserNotificationCenter.current().removePendingNotificationRequests(withIdentifiers: [task.id.uuidString])

        let content = UNMutableNotificationContent()
        content.title = task.title
        content.body = "Don't forget to complete your task!"
        content.sound = .default

        let trigger = UNCalendarNotificationTrigger(dateMatching: Calendar.current.dateComponents(
            [.year, .month, .day, .hour, .minute],
            from: date), repeats: false)

        let request = UNNotificationRequest(identifier: task.id.uuidString, content: content, trigger: trigger)
        UNUserNotificationCenter.current().add(request)
    }

    // MARK: - Overdue Task Check

    private func startContinuousOverdueCheck() {
        Timer.publish(every: 30, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in self?.checkOverdueTasks() }
            .store(in: &cancellables)
    }

    func checkOverdueTasks() {
        let now = Date()

        for task in tasks {
            // Only apply penalty if it hasn't already been applied
            checkAndApplyPenalty(for: task)
        }
    }

    // ✅ NEW: Check and apply overdue penalty for a task
    func checkAndApplyPenalty(for task: Task) {
        guard !task.isCompleted,
              let dueDate = task.dueDate,
              dueDate < Date(),
              !task.hasPenaltyApplied,
              let index = tasks.firstIndex(where: { $0.id == task.id }) else {
            return
        }

        tasks[index].hasPenaltyApplied = true
        hearts = max(0, hearts - 1)

        let generator = UINotificationFeedbackGenerator()
        generator.notificationOccurred(.warning)
    }

    // MARK: - Store Functions

    func buyFish() {
        if coins >= 10 {
            coins -= 10
            fishInventory += 1
        }
    }

    func buyHeart() {
        if coins >= 50 && hearts < 9 {
            coins -= 50
            hearts += 1
        }
    }

    func feedFish() {
        if fishInventory > 0 {
            fishInventory -= 1
            happiness += 10
        }
    }

    // MARK: - Persistence

    private func saveTasks() {
        if let encoded = try? JSONEncoder().encode(tasks) {
            UserDefaults.standard.set(encoded, forKey: tasksKey)
        }
    }

    private func saveData() {
        UserDefaults.standard.set(coins, forKey: coinsKey)
        UserDefaults.standard.set(hearts, forKey: heartsKey)
        UserDefaults.standard.set(happiness, forKey: happinessKey)
        UserDefaults.standard.set(fishInventory, forKey: fishKey)
        UserDefaults.standard.set(highestHeartsMaintained, forKey: "highestHearts")
        UserDefaults.standard.set(daysSurvived, forKey: "daysSurvived")
        UserDefaults.standard.set(lastResetDate, forKey: "lastResetDate")

        if let encoded = try? JSONEncoder().encode(achievements) {
            UserDefaults.standard.set(encoded, forKey: "achievements")
        }
    }

    private func loadData() {
        if let data = UserDefaults.standard.data(forKey: tasksKey),
           let decoded = try? JSONDecoder().decode([Task].self, from: data) {
            tasks = decoded

            // ✅ NEW: Only apply penalties once per day
            let now = Date()
            let lastDeductionDate = UserDefaults.standard.object(forKey: lastDeductionTimestampKey) as? Date ?? .distantPast

            if now.timeIntervalSince(lastDeductionDate) >= 24 * 60 * 60 {
                for task in tasks {
                    checkAndApplyPenalty(for: task)
                }
                UserDefaults.standard.set(now, forKey: lastDeductionTimestampKey)
            }
        }

        if let data = UserDefaults.standard.data(forKey: "achievements"),
           let decoded = try? JSONDecoder().decode([Achievement].self, from: data) {
            achievements = decoded
        }

    }

    // MARK: - Reset System

    private func checkHeartCondition() {
        guard hearts > 0 else {
            performSelectiveReset()
            return
        }

        if hearts > highestHeartsMaintained {
            highestHeartsMaintained = hearts
        }
    }

    private func performSelectiveReset() {
        let currentDays = Calendar.current.dateComponents([.day], from: lastResetDate, to: Date()).day ?? 0
        if currentDays > daysSurvived {
            daysSurvived = currentDays
        }

        DispatchQueue.main.async {
            self.coins = 0
            self.hearts = 9
            self.happiness = 0
            self.fishInventory = 0
            self.lastResetDate = Date()
            self.checkForNewAchievements()
            NotificationCenter.default.post(name: .appDidReset, object: nil)
        }
    }

    // MARK: - Achievements

    func checkForNewAchievements() {
        if daysSurvived >= 7 && !achievements.contains(where: { $0.id == "week_survivor" }) {
            grantAchievement(id: "week_survivor", title: "Week Survivor", description: "Survived 7 days without reset")
        }

        if daysSurvived >= 3 && !achievements.contains(where: { $0.id == "three_day_streak" }) {
            grantAchievement(id: "three_day_streak", title: "Three Day Streak", description: "Logged in for 3 consecutive days")
        }

        if highestHeartsMaintained >= 15 && !achievements.contains(where: { $0.id == "heart_master" }) {
            grantAchievement(id: "heart_master", title: "Heart Master", description: "Maintained 15+ hearts")
        }
    }

    private func grantAchievement(id: String, title: String, description: String) {
        let newAchievement = Achievement(id: id, title: title, description: description, dateEarned: Date())
        achievements.append(newAchievement)
        let generator = UINotificationFeedbackGenerator()
        generator.notificationOccurred(.success)
    }
}
