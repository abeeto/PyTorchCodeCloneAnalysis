import sys
import re
import subprocess
import platform

# Flags to see if the user wants go with the default dark scheme and css
default_color_flag = None
custom_css_flag = None

# The user must input either 1, 5, or 6 arguments
if (len(sys.argv) > 2 and len(sys.argv) < 5) or \
   len(sys.argv) > 6 or len(sys.argv) < 1:
    # Invalid usage
    print("Usage 1: python slack_dark.py")
    print("Usage 2: python slack_dark.py [http/https css link]")
    print("Usage 3: python slack_dark.py [primary] [text] [background] [background_elevated]")
    print("Usage 4: python slack_dark.py [primary] [text] [background] [background_elevated] [http/https css link]")
    print("Note: All colors must be 3 byte hexadecimal HTML color codes in quotes.")
    print("Example: python slack_dark.py \"#FFFFFF\" \"#000000\" \"#101010\" \"#202020\" https://cdn.rawgit.com/widget-/slack-black-theme/master/custom.css")
    exit(1)
elif len(sys.argv) == 1:
    # Just the defaults
    default_color_flag = True
    custom_css = False
elif len(sys.argv) == 2:
    # Defaults with custom css link
    default_color_flag = True
    custom_css = True
elif len(sys.argv) == 5:
    # Custom colors with default css
    default_color_flag = False
    custom_css = False
elif len(sys.argv) == 6:
    # Custom colors with custom css
    default_color_flag = False
    custom_css = True

# If the user wants custom colors, then check to see if they provided the right hex color code
if not default_color_flag:
    pattern = re.compile("^#(?:[0-9a-fA-F]{3}){1,2}$")
    count = 1
    for color_code in sys.argv[1:]:
        if count == 5:
            break
        else:
            count += 1
        if pattern.match(color_code) == None:
            print("Invalid color code: %s" % color_code)
            exit(1)

# If the user wants a custom css, check to verify if it is a correct url. I stole most of this regex not gonna lie.
if custom_css:
    pattern = re.compile("((http|https):\\/\\/)?[\\w\\-_]+(\\.[\\w\\-_]+)+([\\w\\-\\.,@?^=%&amp;:/~\\+#]*[\\w\\-\\@?^=%&amp;/~\\+#])?.css")
    if pattern.match(sys.argv[-1]) == None:
        print("Invalid link to custom_css: %s" % sys.argv[-1])
        print("Must be a http or https link.")
        exit(1)

# Path 1 is for app-2.5.1, Path 2 is for app >= 3.0.0.
# I'm just gonna do both because why the fuck not lol
win_1 = "%homepath%\\AppData\\Local\\slack\\resources\\app.asar.unpacked\\src\\static\\index.js"
win_2 = "%homepath%\\AppData\\Local\\slack\\resources\\app.asar.unpacked\\src\\static\ssb-interop.js"
mac_1 = "/Applications/Slack.app/Contents/resources/app.asar.unpacked/src/static/index.js"
mac_2 = "/Applications/Slack.app/Contents/resources/app.asar.unpacked/src/static/ssb-interop.js"
linux_1 = "/usr/lib/slack/resources/app.asar.unpacked/src/static/index.js"
linux_2 = "/usr/lib/slack/resources/app.asar.unpacked/src/static/ssb-interop.js"

# Load up all paths into a dict to vary based on OS.
all_paths = {
        "Windows": [win_1, win_2],
        "Darwin" : [mac_1, mac_2],
        "Linux"  : [linux_1, linux_2]
        }

# Return the path to be used based off of the platform this is run on.
try:
    cur_path_1, cur_path_2 = all_paths[platform.system()]
except KeyError:
    print(platform.system() + " is not supported.")
    exit(1)

if not default_color_flag:
    # Custom Color Scheme
    primary_color = sys.argv[1]
    text_color = sys.argv[2]
    background_color = sys.argv[3]
    background_elevated_color = sys.argv[4]
else:
    # Default Dark Colors
    primary_color = "#09F"
    text_color = "#CCC"
    background_color = "#080808"
    background_elevated_color = "#222"


# Source: widget-/slack-black-theme (Apache License 2.0)
if custom_css_flag:
    css_link = sys.argv[-1]
else:
    css_link = "https://cdn.rawgit.com/widget-/slack-black-theme/master/custom.css"

# Source: widget-/slack-black-theme & d-fay/slack-black-theme
# Modified a little.
# Event Listener for Slack file
js_code = """
// First make sure the wrapper app is loaded
document.addEventListener("DOMContentLoaded", function() {

   // Then get its webviews
   let webviews = document.querySelectorAll(".TeamView webview");

   // Fetch our CSS in parallel ahead of time
   const cssPath = '""" + css_link + """';
   let cssPromise = fetch(cssPath).then(response => response.text());

   let customCustomCSS = `
   :root {
      /* Modify these to change your theme colors: */
      --primary: """ + primary_color + """;
      --text: """ + text_color + """;
      --background: """ + background_color + """;
      --background-elevated: """ + background_elevated_color + """;
   }
   div.c-message.c-message--light.c-message--hover
   {
   color: #fff !important;
    background-color: """ + background_color + """ !important;
   }

   div.c-virtual_list__scroll_container {
    background-color: """ + background_color + """ !important;
   }
   .p-message_pane .c-message_list:not(.c-virtual_list--scrollbar), .p-message_pane .c-message_list.c-virtual_list--scrollbar > .c-scrollbar__hider {
    z-index: 0;
   }

   div.comment.special_formatting_quote.content,.comment_body{
    color: """ + primary_color + """ !important;
   }

   div.c-message:hover {
    background-color: """ + background_color + """ !important;
   }

   div.c-message_attachment.c-message_attachment{
    color: """ + background_elevated_color + """ !important;
   }

   span.c-message_attachment__pretext{
    color: """ + text_color + """ !important;
   }

   hr.c-message_list__day_divider__line{
    background: """ + background_elevated_color + """ !important;
   }

   div.c-message_list__day_divider__label__pill{
    background: """ + primary_color + """ !important;
   }   

   span.c-message__body,
   a.c-message__sender_link,
   span.c-message_attachment__media_trigger.c-message_attachment__media_trigger--caption,
   div.p-message_pane__foreword__description span
   {
       color: """ + text_color + """ !important;
   }

   pre.special_formatting{
     background-color: """ + background_color + """ !important;
     color: """ + text_color + """ !important;
     border: solid;
     border-width: 1 px !important;
    
   }

   div.ql-editor.c-message__editor__input {
    background: """ + background_color + """ !important;
    color: """ + primary_color + """ !important;
   }
   
   div.c-message--light .c-message--highlight .c-message--editing .c-message--highlight_yellow_bg{
    background: """ + background_color + """ !important;
    border: none !important;
   }

   // Insert a style tag into the wrapper view
   cssPromise.then(css => {
      let s = document.createElement('style');
      s.type = 'text/css';
      s.innerHTML = css + customCustomCSS;
      document.head.appendChild(s);
   });

   // Wait for each webview to load
   webviews.forEach(webview => {
      webview.addEventListener('ipc-message', message => {
         if (message.channel == 'didFinishLoading')
            // Finally add the CSS into the webview
            cssPromise.then(css => {
               let script = `
                     let s = document.createElement('style');
                     s.type = 'text/css';
                     s.id = 'slack-custom-css';
                     s.innerHTML = \\`${css + customCustomCSS}\\`;
                     document.head.appendChild(s);
                     `
               webview.executeJavaScript(script);
            })
      });
   });
});
"""

# Open the target files in append mode and add the code to them.
file_1 = open(cur_path_1, 'a')
file_1.write(js_code + '\n')
file_1.close()

file_2 = open(cur_path_2, 'a')
file_2.write(js_code + '\n')
file_2.close()
