# -*- coding: utf-8 -*- 

###########################################################################
## Python code generated with wxFormBuilder (version Mar 22 2011)
## http://www.wxformbuilder.org/
##
## PLEASE DO "NOT" EDIT THIS FILE!
###########################################################################

import wx

###########################################################################
## Class TopFrame
###########################################################################

class TopFrame ( wx.Frame ):
	
	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 500,300 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
		
		self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
		self.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_BTNFACE ) )
		
		self.m_menubar1 = wx.MenuBar( 0 )
		self.file_menu = wx.Menu()
		self.edit_profile_menu_item = wx.MenuItem( self.file_menu, wx.ID_ANY, u"&Edit Profile"+ u"\t" + u"F2", wx.EmptyString, wx.ITEM_NORMAL )
		self.file_menu.AppendItem( self.edit_profile_menu_item )
		
		self.file_menu.AppendSeparator()
		
		self.open_profile_menu_item = wx.MenuItem( self.file_menu, wx.ID_ANY, u"&Open Profile"+ u"\t" + u"Ctrl+O", wx.EmptyString, wx.ITEM_NORMAL )
		self.file_menu.AppendItem( self.open_profile_menu_item )
		
		self.save_profile_menu_item = wx.MenuItem( self.file_menu, wx.ID_ANY, u"&Save Profile"+ u"\t" + u"Ctrl+S", wx.EmptyString, wx.ITEM_NORMAL )
		self.file_menu.AppendItem( self.save_profile_menu_item )
		
		self.file_menu.AppendSeparator()
		
		self.exit_menu_item = wx.MenuItem( self.file_menu, wx.ID_ANY, u"E&xit"+ u"\t" + u"Alt+F4", wx.EmptyString, wx.ITEM_NORMAL )
		self.file_menu.AppendItem( self.exit_menu_item )
		
		self.m_menubar1.Append( self.file_menu, u"&File" ) 
		
		self.SetMenuBar( self.m_menubar1 )
		
		bSizer1 = wx.BoxSizer( wx.VERTICAL )
		
		self.profile_panel = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.FULL_REPAINT_ON_RESIZE )
		self.profile_panel.SetMinSize( wx.Size( 500,300 ) )
		
		bSizer1.Add( self.profile_panel, 1, wx.ALL|wx.EXPAND, 5 )
		
		self.m_staticline1 = wx.StaticLine( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
		bSizer1.Add( self.m_staticline1, 0, wx.ALL|wx.EXPAND, 0 )
		
		bSizer2 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_staticText1 = wx.StaticText( self, wx.ID_ANY, u"Temp", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText1.Wrap( -1 )
		self.m_staticText1.SetFont( wx.Font( 16, 70, 90, 90, False, wx.EmptyString ) )
		
		bSizer2.Add( self.m_staticText1, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		self.temp_ctrl = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 80,-1 ), wx.TE_CENTRE|wx.TE_READONLY )
		self.temp_ctrl.SetFont( wx.Font( 16, 70, 90, 90, False, wx.EmptyString ) )
		self.temp_ctrl.SetForegroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_CAPTIONTEXT ) )
		self.temp_ctrl.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_3DDKSHADOW ) )
		
		bSizer2.Add( self.temp_ctrl, 0, wx.ALIGN_CENTER|wx.ALL, 5 )
		
		self.start_stop_button = wx.Button( self, wx.ID_ANY, u"Start", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.start_stop_button.SetDefault() 
		self.start_stop_button.SetFont( wx.Font( 16, 70, 90, 92, False, wx.EmptyString ) )
		
		bSizer2.Add( self.start_stop_button, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )
		
		self.m_staticText2 = wx.StaticText( self, wx.ID_ANY, u"Time", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText2.Wrap( -1 )
		self.m_staticText2.SetFont( wx.Font( 16, 70, 90, 90, False, wx.EmptyString ) )
		
		bSizer2.Add( self.m_staticText2, 0, wx.ALIGN_CENTER|wx.ALL, 5 )
		
		self.time_ctrl = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 80,-1 ), wx.TE_CENTRE|wx.TE_READONLY )
		self.time_ctrl.SetFont( wx.Font( 16, 70, 90, 90, False, wx.EmptyString ) )
		self.time_ctrl.SetForegroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_CAPTIONTEXT ) )
		self.time_ctrl.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_3DDKSHADOW ) )
		
		bSizer2.Add( self.time_ctrl, 0, wx.ALIGN_CENTER|wx.ALL, 5 )
		
		bSizer1.Add( bSizer2, 0, wx.ALIGN_CENTER, 5 )
		
		self.SetSizer( bSizer1 )
		self.Layout()
		self.status_bar = self.CreateStatusBar( 1, wx.ST_SIZEGRIP, wx.ID_ANY )
		
		self.Centre( wx.BOTH )
		
		# Connect Events
		self.Bind( wx.EVT_CLOSE, self.on_close )
		self.Bind( wx.EVT_MENU, self.on_edit_profile, id = self.edit_profile_menu_item.GetId() )
		self.Bind( wx.EVT_MENU, self.on_open_profile, id = self.open_profile_menu_item.GetId() )
		self.Bind( wx.EVT_MENU, self.on_save_profile, id = self.save_profile_menu_item.GetId() )
		self.Bind( wx.EVT_MENU, self.on_exit, id = self.exit_menu_item.GetId() )
		self.profile_panel.Bind( wx.EVT_LEAVE_WINDOW, self.on_leave_profile_panel )
		self.profile_panel.Bind( wx.EVT_MOTION, self.on_motion_profile_panel )
		self.profile_panel.Bind( wx.EVT_PAINT, self.on_paint_profile_panel )
		self.profile_panel.Bind( wx.EVT_SIZE, self.on_size_profile_panel )
		self.start_stop_button.Bind( wx.EVT_BUTTON, self.on_start_stop )
	
	def __del__( self ):
		pass
	
	
	# Virtual event handlers, overide them in your derived class
	def on_close( self, event ):
		event.Skip()
	
	def on_edit_profile( self, event ):
		event.Skip()
	
	def on_open_profile( self, event ):
		event.Skip()
	
	def on_save_profile( self, event ):
		event.Skip()
	
	def on_exit( self, event ):
		event.Skip()
	
	def on_leave_profile_panel( self, event ):
		event.Skip()
	
	def on_motion_profile_panel( self, event ):
		event.Skip()
	
	def on_paint_profile_panel( self, event ):
		event.Skip()
	
	def on_size_profile_panel( self, event ):
		event.Skip()
	
	def on_start_stop( self, event ):
		event.Skip()
	

###########################################################################
## Class ProfileConfigDialog
###########################################################################

class ProfileConfigDialog ( wx.Dialog ):
	
	def __init__( self, parent ):
		wx.Dialog.__init__ ( self, parent, id = wx.ID_ANY, title = u"Profile Configuration", pos = wx.DefaultPosition, size = wx.DefaultSize, style = wx.CAPTION )
		
		self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
		
		bSizer5 = wx.BoxSizer( wx.VERTICAL )
		
		self.profile_sizer = wx.FlexGridSizer( 0, 5, 0, 0 )
		self.profile_sizer.SetFlexibleDirection( wx.BOTH )
		self.profile_sizer.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
		
		bSizer5.Add( self.profile_sizer, 1, wx.EXPAND, 5 )
		
		bSizer6 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.apply_button = wx.Button( self, wx.ID_ANY, u"Apply", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.apply_button.SetDefault() 
		bSizer6.Add( self.apply_button, 0, wx.ALL, 5 )
		
		self.copy_all_button = wx.Button( self, wx.ID_ANY, u"Copy to All", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer6.Add( self.copy_all_button, 0, wx.ALL, 5 )
		
		self.cancel_button = wx.Button( self, wx.ID_ANY, u"Cancel", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer6.Add( self.cancel_button, 0, wx.ALL, 5 )
		
		bSizer5.Add( bSizer6, 0, wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.SetSizer( bSizer5 )
		self.Layout()
		bSizer5.Fit( self )
		
		self.Centre( wx.BOTH )
		
		# Connect Events
		self.apply_button.Bind( wx.EVT_BUTTON, self.on_apply_button )
		self.copy_all_button.Bind( wx.EVT_BUTTON, self.on_copy_all_button )
		self.cancel_button.Bind( wx.EVT_BUTTON, self.on_cancel_button )
	
	def __del__( self ):
		pass
	
	
	# Virtual event handlers, overide them in your derived class
	def on_apply_button( self, event ):
		event.Skip()
	
	def on_copy_all_button( self, event ):
		event.Skip()
	
	def on_cancel_button( self, event ):
		event.Skip()
	

###########################################################################
## Class ProfileEntryPanel
###########################################################################

class ProfileEntryPanel ( wx.Panel ):
	
	def __init__( self, parent ):
		wx.Panel.__init__ ( self, parent, id = wx.ID_ANY, pos = wx.DefaultPosition, size = wx.Size( 210,40 ), style = wx.TAB_TRAVERSAL )
		
		self.SetMinSize( wx.Size( 210,40 ) )
		self.SetMaxSize( wx.Size( 210,40 ) )
		
		bSizer4 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.entry_index_text = wx.StaticText( self, wx.ID_ANY, u"Entry 0", wx.DefaultPosition, wx.Size( 45,-1 ), wx.ALIGN_CENTRE|wx.ST_NO_AUTORESIZE )
		self.entry_index_text.Wrap( -1 )
		bSizer4.Add( self.entry_index_text, 0, wx.ALIGN_CENTER|wx.LEFT|wx.RIGHT, 5 )
		
		self.temperature_ctrl = wx.TextCtrl( self, wx.ID_ANY, u"15 (C)", wx.DefaultPosition, wx.Size( 60,-1 ), wx.TE_CENTRE )
		bSizer4.Add( self.temperature_ctrl, 0, wx.ALIGN_CENTER|wx.LEFT|wx.RIGHT, 5 )
		
		self.time_ctrl = wx.TextCtrl( self, wx.ID_ANY, u"10 s", wx.DefaultPosition, wx.Size( 60,-1 ), wx.TE_CENTRE )
		bSizer4.Add( self.time_ctrl, 0, wx.ALIGN_CENTER|wx.LEFT|wx.RIGHT, 5 )
		
		self.SetSizer( bSizer4 )
		self.Layout()
		
		# Connect Events
		self.temperature_ctrl.Bind( wx.EVT_KILL_FOCUS, self.on_kill_focus_temp )
		self.temperature_ctrl.Bind( wx.EVT_SET_FOCUS, self.on_set_focus_temp )
		self.time_ctrl.Bind( wx.EVT_KILL_FOCUS, self.on_kill_focus_time )
		self.time_ctrl.Bind( wx.EVT_SET_FOCUS, self.on_set_focus_time )
	
	def __del__( self ):
		pass
	
	
	# Virtual event handlers, overide them in your derived class
	def on_kill_focus_temp( self, event ):
		event.Skip()
	
	def on_set_focus_temp( self, event ):
		event.Skip()
	
	def on_kill_focus_time( self, event ):
		event.Skip()
	
	def on_set_focus_time( self, event ):
		event.Skip()
	

