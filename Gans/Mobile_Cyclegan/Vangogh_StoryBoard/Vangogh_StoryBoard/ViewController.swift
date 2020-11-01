//
//  ViewController.swift
//  Vangogh_StoryBoard
//
//  Created by Filip Haltmayer on 10/31/20.
//

import UIKit

class ViewController: UIViewController, UIPickerViewDelegate, UIPickerViewDataSource {

    static var myString = String()
    
    
    @IBOutlet weak var picker: UIPickerView!

    var pickerData: [String] = ["192x144", "352x288", "480x360", "640x480", "1280x720"]


    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.picker.delegate = self
        self.picker.dataSource = self
        // Do any additional setup after loading the view.

    }
    
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1
    }
    
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return pickerData.count
    }
    
    func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
        return pickerData[row]
    }
    
    func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
        ViewController.myString = String(pickerData[row])
    }
}

