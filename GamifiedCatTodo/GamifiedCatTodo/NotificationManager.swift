//
//  NotificationManager.swift
//  GamifiedCatTodo
//
//  Created by Kwan Yi Yan on 7/5/2025.
//
import UserNotifications
import os.log

class NotificationManager {
    static let shared = NotificationManager()
    private let logger = Logger(subsystem: "com.your.app", category: "Notifications")

    func requestAuthorization() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { granted, error in
            if granted {
                self.logger.info("Notification permission granted")
            } else if let error = error {
                self.logger.error("Notification permission error: \(error.localizedDescription)")
            }
        }
    }

    func scheduleNotification(for task: Task) {
        guard let notifyDate = task.notificationDate ?? task.dueDate?.addingTimeInterval(-3600),
              notifyDate > Date() else {
            logger.debug("No valid notification date for task \(task.title)")
            return
        }

        let content = UNMutableNotificationContent()
        content.title = "🔔 \(task.title)"
        content.body = "Due \(task.dueDate?.formatted() ?? "soon")"
        content.sound = .default
        content.userInfo = ["taskID": task.id.uuidString]

        let triggerComponents = Calendar.current.dateComponents(
            [.year, .month, .day, .hour, .minute],
            from: notifyDate
        )

        let trigger = UNCalendarNotificationTrigger(
            dateMatching: triggerComponents,
            repeats: false
        )

        let request = UNNotificationRequest(
            identifier: task.id.uuidString,
            content: content,
            trigger: trigger
        )

        UNUserNotificationCenter.current().add(request)
    }

    func cancelNotification(for taskID: UUID) {
        UNUserNotificationCenter.current().removePendingNotificationRequests(withIdentifiers: [taskID.uuidString])
    }

    func rescheduleAllNotifications(for tasks: [Task]) {
        UNUserNotificationCenter.current().removeAllPendingNotificationRequests()

        tasks.filter { $0.wantsNotification && !$0.isCompleted }
            .forEach { scheduleNotification(for: $0) }
    }
}
