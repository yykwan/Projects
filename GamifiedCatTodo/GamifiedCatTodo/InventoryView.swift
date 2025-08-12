//
//  InventoryView.swift
//  GamifiedCatTodo
//
//  Created by Kwan Yi Yan on 2/5/2025.
//
import SwiftUI

struct InventoryView: View {
    @ObservedObject var viewModel: TaskViewModel
    @Binding var isPresented: Bool
    
    var body: some View {
        VStack {
            // Header with close button
            HStack {
                Text("Inventory")
                    .font(.largeTitle)
                    .bold()
                    .padding(.leading)
                
                Spacer()
                
                Button(action: {
                    isPresented = false
                }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title)
                        .foregroundColor(.gray)
                }
                .padding(.trailing)
            }
            .padding(.top)
            
            Divider()
                .padding(.horizontal)
            
            // Inventory items
            VStack(spacing: 20) {
                // Fish inventory with visual representation
                HStack(spacing: 15) {
                    ForEach(0..<min(viewModel.fishInventory, 10), id: \.self) { _ in
                        Image(systemName: "fish.fill")
                            .font(.title)
                            .foregroundColor(.blue)
                    }
                    
                    if viewModel.fishInventory > 10 {
                        Text("+\(viewModel.fishInventory - 10)")
                            .font(.headline)
                    }
                }
                .padding()
                .background(Color.blue.opacity(0.1))
                .cornerRadius(15)
                
                Text("\(viewModel.fishInventory) Fish in Inventory")
                    .font(.title2)
                    .foregroundColor(.blue)
                
                // Feed fish button
                Button(action: {
                    viewModel.feedFish()
                }) {
                    HStack {
                        Image(systemName: "fish.fill")
                        Text("Feed Fish")
                    }
                    .font(.title2)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .disabled(viewModel.fishInventory <= 0)
                .opacity(viewModel.fishInventory <= 0 ? 0.6 : 1.0)
            }
            .padding()
            
            Spacer()
            
            // Happiness meter
            VStack {
                Text("Current Happiness")
                    .font(.headline)
                
            
                
                Text("\(viewModel.happiness)%")
                    .font(.title2)
                    .bold()
            }
            .padding()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct InventoryView_Previews: PreviewProvider {
    static var previews: some View {
        InventoryView(viewModel: TaskViewModel(), isPresented: .constant(true))
    }
}
