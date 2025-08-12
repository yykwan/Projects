//
//  StoreView.swift
//  GamifiedCatTodo
//
//  Created by Kwan Yi Yan on 2/5/2025.
//

import SwiftUI

struct StoreView: View {
    @ObservedObject var viewModel: TaskViewModel
    @Binding var isPresented: Bool
    
    var body: some View {
        VStack {
            // Header with close button
            HStack {
                Spacer()
                Button(action: {
                    isPresented = false
                }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title)
                        .foregroundColor(.gray)
                }
                .padding()
            }
            
            Spacer()
            
            // Store items
            VStack(spacing: 20) {
                Button(action: {
                    viewModel.buyFish()
                }) {
                    HStack {
                        Image(systemName: "fish.fill")
                        Text("Buy Fish (10 Coins)")
                    }
                    .font(.title2)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .disabled(viewModel.coins < 10)
                .opacity(viewModel.coins < 10 ? 0.6 : 1.0)
                
                Button(action: {
                    viewModel.buyHeart()
                }) {
                    HStack {
                        Image(systemName: "heart.fill")
                        Text("Buy Heart (50 Coins)")
                    }
                    .font(.title2)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.red)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .disabled(viewModel.coins < 50 || viewModel.hearts >= 9)
                .opacity((viewModel.coins < 50 || viewModel.hearts >= 9) ? 0.6 : 1.0)
            }
            .padding()
            
            Spacer()
            
            // Current balance
            Text("Current Balance: \(viewModel.coins) Coins")
                .font(.headline)
                .padding()
        }
    }
}
