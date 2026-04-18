import signal


def install_sigint_handler(stop_event):
    def on_sigint(*_):
        print("Ctrl+C pressed, stopping...")
        stop_event.set()

    signal.signal(signal.SIGINT, on_sigint)
