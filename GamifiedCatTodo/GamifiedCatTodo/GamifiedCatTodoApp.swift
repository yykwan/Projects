//
//  GamifiedCatTodoApp.swift
//  GamifiedCatTodo
//
//  Created by Kwan Yi Yan on 2/5/2025.
//
import SwiftUI
import UserNotifications

@main
struct GamifiedCatTodoApp: App {
    @AppStorage("isDarkMode") private var isDarkMode = false
    @StateObject var viewModel = TaskViewModel()
    @StateObject private var taskManager = TaskManager()
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    
    init() {
        NotificationManager.shared.requestAuthorization()
        UNUserNotificationCenter.current().delegate = NotificationDelegate.shared
    }

    var body: some Scene {
        WindowGroup {
            ContentView(viewModel: viewModel)
                .preferredColorScheme(isDarkMode ? .dark : .light)
                .environmentObject(taskManager)
                .onReceive(NotificationCenter.default.publisher(for: UIApplication.willEnterForegroundNotification)) { _ in
                    taskManager.rescheduleNotifications()
                }
                .onAppear {
                    // Initial reschedule with slight delay to ensure TaskManager is fully loaded
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        taskManager.rescheduleNotifications()
                    }
                }
        }
    }
}

class NotificationDelegate: NSObject, UNUserNotificationCenterDelegate {
    static let shared = NotificationDelegate()
    
    func userNotificationCenter(_ center: UNUserNotificationCenter,
                              willPresent notification: UNNotification,
                              withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void) {
        completionHandler([.banner, .sound, .badge])
    }
    
    func userNotificationCenter(_ center: UNUserNotificationCenter,
                              didReceive response: UNNotificationResponse,
                              withCompletionHandler completionHandler: @escaping () -> Void) {
        defer { completionHandler() }
        
        guard let taskIDString = response.notification.request.content.userInfo["taskID"] as? String,
              let taskID = UUID(uuidString: taskIDString) else {
            return
        }
        
        NotificationCenter.default.post(
            name: .init("NotificationTapped"),
            object: nil,
            userInfo: ["taskID": taskID]
        )
    }
}

// MARK: - Notification Handling Extensions
extension Notification.Name {
    static let notificationTapped = Notification.Name("NotificationTapped")
}



