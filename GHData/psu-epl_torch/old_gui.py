'''
Created on May 20, 2013

@author: User
'''
import csv
import time
import wx
import wxfb_out
from serial.tools.list_ports import comports as comports
from struct import error as struct_error

import TorchOven

SIMULATE_TORCH = False # Debug variable - enabling this DISABLES communication with the actual Torch!

PROFILE_LENGTH = 40 # From listening to the Torch's controller - might be changeable, *might not*

TIME_AXIS_LABEL_FREQUENCY = 60
TEMP_AXIS_LABEL_FREQUENCY = 50

TEMP_MIN = 1
TEMP_MAX = 300
TEMP_SUFFIX = ' (C)'
TIME_MIN = 1
TIME_MAX = 600
TIME_SUFFIX = ' s'

def clean_limit_string(string, minimum, maximum):
    new_string = ''
    for char in string:
        if char.isdigit():
            new_string += char
    numerical_value = int(new_string)
    if numerical_value < minimum:
        new_string = str(minimum)
    elif numerical_value > maximum:
        new_string = str(maximum)
    return new_string

def clean_limit_append_string(string, minimum, maximum, suffix):
    return clean_limit_string(string, minimum, maximum)+suffix

class TopFrame(wxfb_out.TopFrame):
    def __init__(self):
        super(TopFrame, self).__init__(None)
        self.profile_panel.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.profile = TorchOven.DEFAULT_PROFILE
        self.oven = None
        p_width, p_height = self.profile_panel.GetSizeTuple()
        max_time = 0
        min_temp = 15
        max_temp = 15
        for temp_time_pair in self.profile:
            min_temp = min(min_temp, temp_time_pair[0])
            max_temp = max(max_temp, temp_time_pair[0])
            max_time += temp_time_pair[1]
        self.total_time = max_time
        self.view = View(p_width, p_height, max_time, min_temp, max_temp)
        self.actual_readings = [] # To be populated by tuples of (temperature, current time-t=0 of reading)
        self.current_time = None
        self.started = False
        self.ideal_profile_pen = wx.Pen('green', 2)
        self.actual_profile_pen = wx.Pen('red', 2)
        self.start_stop_button.SetForegroundColour('green')
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_timer)
        self.Fit()
        self.SetMinSize(self.GetSize())
        self.Maximize()
        self.Center()

    def parse_new_profile(self, input_profile):
        new_profile = input_profile
        return new_profile

    def on_leave_profile_panel(self, event):
        self.status_bar.SetStatusText('')

    def on_motion_profile_panel(self, event):
        time = self.view.x_p2l(event.GetX(), True)
        temp = self.view.y_p2l(event.GetY(), True)
        time_string = str(time)+' seconds' if time else ''
        temp_string = str(temp)+' (C)' if temp else ''
        if time_string:
            total_string = time_string
            if temp_string:
                total_string += ', '+temp_string
        elif temp_string:
            total_string = temp_string
        else:
            total_string = ''
        self.status_bar.SetStatusText(total_string)

    def on_paint_profile_panel(self, event):
        dc = wx.AutoBufferedPaintDC(self.profile_panel)
        dc.Clear()
        dc.SetBrush(wx.BLACK_BRUSH)
        max_time_text_height = 0
        max_temp_text_width = 0
        for time_temp_pair in self.profile:
            time_text_height = dc.GetTextExtent(str(time_temp_pair[0]))[1]
            if time_text_height > max_time_text_height:
                max_time_text_height = time_text_height
            temp_text_width = dc.GetTextExtent(str(time_temp_pair[1]))[0]
            if temp_text_width > max_temp_text_width:
                max_temp_text_width = temp_text_width
        self.view.p_x_text_border = max_temp_text_width
        self.view.p_y_text_border = max_time_text_height
        self.view.calc_scales()
        flood_fill_x = int(self.view.p_width/2)
        flood_fill_y = int(self.view.p_height/2)
        dc.FloodFill(flood_fill_x, flood_fill_y, 'white')
        dc.SetTextForeground('white')
        dc.SetPen(wx.WHITE_PEN)
        next_time = 0
        while next_time < self.view.max_time:
            string_label = str(next_time)
            text_width, text_height = dc.GetTextExtent(string_label)
            x_pos = self.view.x_l2p(next_time)
            y_pos = self.view.p_height-text_height
            dc.DrawText(string_label, round(x_pos-text_width/2), y_pos)
            dc.DrawLine(x_pos, 0, x_pos, self.view.p_height-self.view.p_y_text_border)
            next_time += TIME_AXIS_LABEL_FREQUENCY
        next_temp = 0
        while next_temp < self.view.max_temp:
            string_label = str(next_temp)
            text_width, text_height = dc.GetTextExtent(string_label)
            y_pos = self.view.y_l2p(next_temp)
            dc.DrawText(string_label, self.view.p_x_text_border-text_width, round(y_pos-text_height/2))
            dc.DrawLine(self.view.p_x_text_border, y_pos, self.view.p_width, y_pos)
            next_temp += TEMP_AXIS_LABEL_FREQUENCY
        dc.SetPen(self.ideal_profile_pen)
        last_x = self.view.p_x_text_border
        running_time = 0
        for temp_time_pair in self.profile:
            running_time += temp_time_pair[1]
            new_x = self.view.x_l2p(running_time)
            y = self.view.y_l2p(temp_time_pair[0])
            dc.DrawLine(last_x, y, new_x, y)
            last_x = new_x
        if self.current_time != None and self.actual_readings:
            dc.SetPen(self.actual_profile_pen)
            iterator_x = self.view.x_l2p(self.view.min_time)
            current_x = self.view.x_l2p(self.current_time)
            running_time = 0
            last_found_temp_index = 0
            temp_list_length = len(self.actual_readings)
            while iterator_x <= current_x:
                if last_found_temp_index < temp_list_length-1:
                    if self.view.x_p2l(iterator_x) >= self.actual_readings[last_found_temp_index+1][1]:
                        last_found_temp_index += 1
                dc.DrawPoint(iterator_x, self.view.y_l2p(self.actual_readings[last_found_temp_index][0]))
                iterator_x += 1

    def on_size_profile_panel(self, event):
        p_width, p_height = self.profile_panel.GetSizeTuple()
        self.view.set_p(p_width, p_height)
        self.view.calc_scales()

    def on_edit_profile(self, event):
        dlg = ProfileConfigDialog(self, self.profile)
        if dlg.ShowModal():
            self.profile = dlg.output_profile
            max_time = 0
            for entry in self.profile:
                max_time += entry[1]
            self.total_time = max_time
            self.profile_panel.Refresh()

    def on_open_profile(self, event):
        profile_filename = wx.FileSelector('Choose a PRFL file', '.', wildcard = '*.prfl',
                                           flags = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if profile_filename:
            loaded_profile = []
            # modified by Pat 2013-12-02 to not use csv dictreader.
            # This is simpler and allows for comment lines in the profile.
            lines = [L.strip().upper().split(',') for L in open(profile_filename,'r').readlines() if L.strip() and (L.strip()[0] != '#')]
            if lines[0][0] != 'TEMP' or lines[0][1] != 'TIME':
                wx.MessageBox('First non-comment line is not Temp, Time header')
                return
            for L in lines[1:]: # skip header line
                loaded_profile.append((int(L[0]), int(L[1])))
            if len(loaded_profile) < PROFILE_LENGTH:
                loaded_profile = loaded_profile[:PROFILE_LENGTH]
            while len(loaded_profile) < PROFILE_LENGTH:
                loaded_profile.append((0, 5))
            self.profile = loaded_profile
#             f = open(profile_filename,'r')  
#             csvfile = csv.DictReader(f)
#             found_rows = 0
#             last_found_temp = None
#             for row in csvfile:
#                 if 'Temp' in row and 'Time' in row:
#                     last_found_temp = int(clean_limit_string(row['Temp'], TEMP_MIN, TEMP_MAX))
#                     loaded_profile.append((last_found_temp,
#                                            int(clean_limit_string(row['Time'], TIME_MIN, TIME_MAX))
#                                          ))
#                     found_rows += 1
#                     if found_rows == PROFILE_LENGTH:
#                         break
#             if len(loaded_profile) < PROFILE_LENGTH:
#                 if last_found_temp:
#                     loaded_profile.append((last_found_temp, 1) * (PROFILE_LENGTH-len(loaded_profile)))
#                 else:
#                     wx.MessageBox('Unable to read temperatures from file', 'File parsing error')
#                     return False

            self.Refresh()
            return True

    def on_save_profile(self, event):
        profile_filename = wx.FileSelector('Select or enter a filename', '.', wildcard = '*.prfl',
                                           flags = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if profile_filename:
            f = open(profile_filename, 'w')
            f.write('Temp,Time')
            for entry in self.profile:
                f.write('\n'+str(entry[0])+','+str(entry[1]))

    def on_start_stop(self, event):
        if self.started:
            self.stop_torch()
        else:
            self.edit_profile_menu_item.Enable(False)
            self.open_profile_menu_item.Enable(False)
            self.current_time = None
            self.actual_readings = []
            if SIMULATE_TORCH:
                self.oven = TorchOven.VirtualTorchOven()
                attempt = True
            else:
                ports = [cp[0] for cp in comports()]
                if ports:
                    if len(ports) > 1:
                        user_choice = wx.GetSingleChoiceIndex('Choose which COM port the Torch is connected to', 'Choose port', ports)
                        if user_choice == -1:
                            attempt = False
                        else:
                            target_port = ports[user_choice]
                            attempt = True
                    else:
                        target_port = ports[0]
                        attempt = True
                else:
                    wx.MessageBox('No COM ports were enumerated by the comports generator', 'No ports detected')
                if attempt:
                    self.oven = TorchOven.TorchOven(target_port)
            if attempt:
                self.oven.init_sequence()
                self.oven.send_profile(self.profile)
                try:
                    self.oven.read_profile()
                except struct_error:
                    pass
                self.oven.start()
                self.started = time.time()
                self.current_time = 0
                self.temp_read_failures = 0
                self.timer.Start(1000)
                self.start_stop_button.SetLabel('Stop')
                self.start_stop_button.SetForegroundColour('red')
                self.profile_panel.Refresh()

    def stop_torch(self, label_niceties = True):
        self.oven.stop()
        self.oven.close()
        self.oven = None
        self.started = False
        self.timer.Stop()
        if label_niceties:
            self.start_stop_button.SetLabel('Start')
            self.start_stop_button.SetForegroundColour('green')
            self.edit_profile_menu_item.Enable()
            self.open_profile_menu_item.Enable()

    def on_timer(self, event):
        current_dtime = time.time() - self.started
        self.current_time = int(current_dtime)
        try:
            new_temp = self.oven.read_temp()
            self.temp_read_failures = 0
            self.actual_readings.append((new_temp, current_dtime))
        except struct_error:
            if self.temp_read_failures < 4:
                self.temp_read_failures += 1
            new_temp = 'ERR'
        self.temp_ctrl.ChangeValue(str(new_temp))
        self.time_ctrl.ChangeValue(str(self.current_time))
        if self.current_time > self.total_time:
            self.stop_torch()
            wx.MessageBox('Reflow cycle complete', 'Cycle complete')
        self.profile_panel.Refresh()

    def on_close(self, event):
        if event.CanVeto():
            if self.oven:
                if wx.MessageBox('Are you sure you want to close and cancel the current reflow cycle?', 'Confirm exit', wx.YES_NO) == wx.YES:
                    self.stop_torch(False)
                    self.Destroy()
            else:
                self.Destroy()
        else:
            self.Destroy()

    def on_exit(self, event):
        self.Close()

class View(object):
    def __init__(self, p_width, p_height, max_time, min_temp, max_temp):
        self.p_x_text_border = 0
        self.p_y_text_border = 0
        self.set_p(p_width, p_height)
        self.min_time = 0
        self.max_time = max_time
        self.min_temp = min(min_temp, 25) - 10
        self.max_temp = max_temp + 10
        self.calc_l()
        self.calc_scales()

    def x_l2p(self, time):
        return int(round((time-self.min_time)*self.x_scale))+self.p_x_text_border

    def y_l2p(self, temp):
        return self.p_height - (int(round((temp-self.min_temp)*self.y_scale))+self.p_y_text_border)

    def x_p2l(self, x_pos, check_boundary = False):
        if check_boundary:
            if x_pos < self.p_x_text_border:
                return False
        return int(round((x_pos-self.p_x_text_border)/self.x_scale + self.min_time))

    def y_p2l(self, y_pos, check_boundary = False):
        if check_boundary:
            if y_pos > self.p_height - self.p_y_text_border:
                return False
        return int(round((self.p_height - (y_pos+self.p_y_text_border))/self.y_scale + self.min_temp))

    def calc_l(self):
        self.l_width = self.max_time - self.min_time
        self.l_height = self.max_temp - self.min_temp

    def set_p(self, width, height):
        self.p_width = width
        self.p_height = height

    def calc_scales(self):
        try:
            self.x_scale = float(self.p_width - self.p_x_text_border) / float(self.l_width)
            self.y_scale = float(self.p_height - self.p_y_text_border) / float(self.l_height)
            return True
        except:
            return False

class ProfileConfigDialog(wxfb_out.ProfileConfigDialog):
    def __init__(self, parent, profile):
        super(ProfileConfigDialog, self).__init__(parent)
        self.entry_panels = []
        for i in xrange(len(profile)):
            new_entry_panel = ProfileEntryPanel(self, i+1, profile[i], self.on_edit_callback)
            self.profile_sizer.Insert(i, new_entry_panel, 0, wx.ALIGN_CENTER)
            self.entry_panels.append(new_entry_panel)
        self.last_edited = None
        self.Fit()
        self.Center()

    def on_edit_callback(self, active_panel):
        self.last_edited = active_panel

    def on_copy_all_button(self, event):
        if self.last_edited:
            active_item_index = self.entry_panels.index(self.last_edited)
            if active_item_index != len(self.entry_panels)-1:
                temp_to_copy = self.last_edited.temperature_ctrl.GetValue()
                time_to_copy = '1'+TIME_SUFFIX
                for item in self.entry_panels[active_item_index+1:]:
                    item.temperature_ctrl.ChangeValue(temp_to_copy)
                    item.time_ctrl.ChangeValue(time_to_copy)

    def on_apply_button(self, event):
        self.output_profile = []
        for entry_panel in self.entry_panels:
            self.output_profile.append(entry_panel.get_value_tuple())
        self.EndModal(1)

    def on_cancel_button(self, event):
        self.EndModal(0)

class ProfileEntryPanel(wxfb_out.ProfileEntryPanel):
    def __init__(self, parent, entry_index, temp_time_tuple, edit_callback):
        super(ProfileEntryPanel, self).__init__(parent)
        self.entry_index_text.SetLabel('Entry '+str(entry_index))
        self.temperature_ctrl.ChangeValue(str(temp_time_tuple[0])+TEMP_SUFFIX)
        self.time_ctrl.ChangeValue(str(temp_time_tuple[1])+TIME_SUFFIX)
        self.edit_callback = edit_callback

    def get_value_tuple(self):
        return (int(self.temperature_ctrl.GetValue().split(TEMP_SUFFIX)[0]), int(self.time_ctrl.GetValue().split(TIME_SUFFIX)[0]))

    def on_kill_focus_temp(self, event):
        target_ctrl = event.GetEventObject()
        if target_ctrl.IsModified:
            self.edit_callback(self)
        target_ctrl.ChangeValue(clean_limit_append_string(target_ctrl.GetValue(), TEMP_MIN, TEMP_MAX, TEMP_SUFFIX))

    def on_set_focus_temp(self, event):
        target_ctrl = event.GetEventObject()
        target_ctrl.ChangeValue(target_ctrl.GetValue().split(TEMP_SUFFIX)[0])

    def on_kill_focus_time(self, event):
        target_ctrl = event.GetEventObject()
        if target_ctrl.IsModified:
            self.edit_callback(self)
        target_ctrl.ChangeValue(clean_limit_append_string(target_ctrl.GetValue(), TIME_MIN, TIME_MAX, TIME_SUFFIX))

    def on_set_focus_time(self, event):
        target_ctrl = event.GetEventObject()
        target_ctrl.ChangeValue(target_ctrl.GetValue().split(TIME_SUFFIX)[0])

if __name__=='__main__':
    app = wx.PySimpleApp()
    f = TopFrame()
    app.SetTopWindow(f)
    f.Show()
    app.MainLoop()