//
//  Task.swift
//  GamifiedCatTodo
//
//  Created by Kwan Yi Yan on 2/5/2025.
import Foundation
import SwiftUI

enum TaskPriority: String, Codable, CaseIterable {
    case low = "Low"
    case medium = "Medium"
    case high = "High"
}

class Task: ObservableObject, Identifiable, Codable {
    let id: UUID

    @Published var title: String
    @Published var category: String
    @Published var priority: TaskPriority
    @Published var dueDate: Date?
    @Published var isCompleted: Bool
    @Published var isPinned: Bool
    @Published var hasPenaltyApplied: Bool = false
    @Published var wantsNotification: Bool
    @Published var notificationDate: Date?

    init(
        id: UUID = UUID(),
        title: String,
        category: String,
        priority: TaskPriority,
        isCompleted: Bool = false,
        isPinned: Bool = false,
        dueDate: Date? = nil,
        wantsNotification: Bool = false,
        notificationDate: Date? = nil
    ) {
        self.id = id
        self.title = title
        self.category = category
        self.priority = priority
        self.isCompleted = isCompleted
        self.isPinned = isPinned
        self.dueDate = dueDate
        self.hasPenaltyApplied = false
        self.wantsNotification = wantsNotification
        self.notificationDate = notificationDate
    }

    var shouldScheduleNotification: Bool {
        wantsNotification && effectiveReminderTime() != nil
    }

    var defaultReminderTime: Date? {
        dueDate?.addingTimeInterval(-3600) // Default reminder 1 hour before due date
    }

    func effectiveReminderTime() -> Date? {
        notificationDate ?? defaultReminderTime
    }

    var isOverdue: Bool {
        guard let dueDate = dueDate else { return false }
        return !isCompleted && dueDate < Date()
    }

    var dueDateString: String {
        guard let dueDate = dueDate else { return "No due date" }
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: dueDate)
    }

    // MARK: - Penalty Check & Application
    // Apply penalty if the task is overdue and no penalty has been applied
    var applyPenaltyIfOverdue: Bool {
        if isOverdue && !hasPenaltyApplied {
            return true // Return true to indicate penalty should be applied
        }
        return false
    }

    // This method will apply penalty and mark the task as having a penalty applied
    func checkAndApplyPenalty() {
        if applyPenaltyIfOverdue {
            // Apply penalty logic here (e.g., deducting hearts)
            self.hasPenaltyApplied = true
            // Persist the penalty applied state to ensure it's not re-applied after restart
            save()
        }
    }

    // MARK: - Saving Task State to UserDefaults
    func save() {
        // Ensure task is saved with updated penalty state
        if let encoded = try? JSONEncoder().encode(self) {
            UserDefaults.standard.set(encoded, forKey: "task-\(id.uuidString)")
        }
    }

    // MARK: - Codable conformance
    enum CodingKeys: CodingKey {
        case id, title, category, priority, dueDate, isCompleted, isPinned, hasPenaltyApplied, wantsNotification, notificationDate
    }

    required convenience init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let id = try container.decode(UUID.self, forKey: .id)
        let title = try container.decode(String.self, forKey: .title)
        let category = try container.decode(String.self, forKey: .category)
        let priority = try container.decode(TaskPriority.self, forKey: .priority)
        let dueDate = try container.decodeIfPresent(Date.self, forKey: .dueDate)
        let isCompleted = try container.decode(Bool.self, forKey: .isCompleted)
        let isPinned = try container.decode(Bool.self, forKey: .isPinned)
        let hasPenaltyApplied = try container.decode(Bool.self, forKey: .hasPenaltyApplied)
        let wantsNotification = try container.decode(Bool.self, forKey: .wantsNotification)
        let notificationDate = try container.decodeIfPresent(Date.self, forKey: .notificationDate)

        self.init(id: id, title: title, category: category, priority: priority,
                  isCompleted: isCompleted, isPinned: isPinned, dueDate: dueDate,
                  wantsNotification: wantsNotification, notificationDate: notificationDate)
        self.hasPenaltyApplied = hasPenaltyApplied
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(title, forKey: .title)
        try container.encode(category, forKey: .category)
        try container.encode(priority, forKey: .priority)
        try container.encode(dueDate, forKey: .dueDate)
        try container.encode(isCompleted, forKey: .isCompleted)
        try container.encode(isPinned, forKey: .isPinned)
        try container.encode(hasPenaltyApplied, forKey: .hasPenaltyApplied)
        try container.encode(wantsNotification, forKey: .wantsNotification)
        try container.encode(notificationDate, forKey: .notificationDate)
    }

    // MARK: - Notification Handling
    func scheduleNotification() {
        guard effectiveReminderTime() != nil else { return }
        
        // If you plan to handle notification scheduling here, you'd interact with UNUserNotificationCenter
        // You'd need the appropriate code to schedule a notification for this task
    }
    
    func cancelNotification() {
        // Logic to cancel the task's scheduled notification
        // Typically, you'd cancel the notification in TaskViewModel or related notification handler
    }
}
