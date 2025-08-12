//
//  AppDelegate.swift
//  GamifiedCatTodo
//
//  Created by Kwan Yi Yan on 7/5/2025.
//
import UIKit
import UserNotifications
import SwiftUI // Import SwiftUI for UIHostingController


class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?
    var taskViewModel = TaskViewModel() // Instantiate the TaskViewModel

    // MARK: - Application Lifecycle
    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        // Setup Notification Center
        UNUserNotificationCenter.current().delegate = self
        
        // Request permission for notifications
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound]) { granted, error in
            if granted {
                print("Notification permission granted.")
            } else {
                print("Notification permission denied.")
            }
        }
        
        // You can also setup the root view controller if you're using SwiftUI
        window = UIWindow(frame: UIScreen.main.bounds)
        
        // Pass taskViewModel to ContentView
        let contentView = ContentView(viewModel: taskViewModel)
        
        // Create UIHostingController with the viewModel passed
        let hostingController = UIHostingController(rootView: contentView)
        
        window?.rootViewController = hostingController
        window?.makeKeyAndVisible()
        
        return true
    }

    // MARK: - Notification Management
    func application(_ application: UIApplication, didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data) {
        // If using push notifications, configure device token here
        // Push notification handling can be added
    }
    
    // Handling notification when app is in background or terminated
    func application(_ application: UIApplication, didReceiveRemoteNotification userInfo: [AnyHashable: Any], fetchCompletionHandler completionHandler: @escaping (UIBackgroundFetchResult) -> Void) {
        // Handle background fetch when a push notification is received
        completionHandler(.newData)
    }
}

extension AppDelegate: UNUserNotificationCenterDelegate {
    // Handle notifications when the app is in the foreground
    func userNotificationCenter(_ center: UNUserNotificationCenter, willPresent notification: UNNotification) async -> UNNotificationPresentationOptions {
        return [.banner, .sound]
    }

    // Handle notification tap
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse,
        withCompletionHandler completionHandler: @escaping () -> Void
    ) {
        // Handle the tapped notification response here if needed
        completionHandler()
    }
}
