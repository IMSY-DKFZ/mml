# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

"""
Module to send notifications of MML execution via various media to the user.
"""

import datetime
import email.message
import json
import logging
import os
import smtplib
import socket
import ssl
import traceback
import warnings
from abc import ABC, abstractmethod
from typing import Optional

import requests
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import mml
from mml.core.data_loading.file_manager import MMLFileManager

logger = logging.getLogger(__name__)


class BaseNotifier(ABC):
    def __init__(self, on_start: bool = False, on_end: bool = False, on_failure: bool = True):
        """
        The notifier base class provides utilities to send messages for mml events.

        :param on_start: emit message on mml start
        :param on_end: emit message on (successful) mml end
        :param on_failure: emit message on mml failure (except keyboard interrupt / user caused SIGTERM signal)
        """
        self.do_on_start = on_start
        self.do_on_end = on_end
        self.do_on_failure = on_failure

    @abstractmethod
    def emit_message(self, text: str) -> None:
        """
        Abstract method that must implement the message delivery mechanism for inherited classes.

        :param text: the text to be sent
        :return:
        """
        pass

    @staticmethod
    def is_master() -> bool:
        """
        Checks if the current process is the master process - used e.g. in Multi-GPU settings to prevent sending
        multiple messages from each node / process.

        :return:
        """
        if "RANK" in os.environ:
            master_process = int(os.environ["RANK"]) == 0
        else:
            master_process = True
        return master_process

    def notify_on_failure(self, error: BaseException) -> None:
        """
        The notification wrapper function to be called from outside in case MML fails.

        :param Exception error: an error that caused MML to fail
        :return:
        """
        # skip notification if either not requested or not on master node
        if not (self.do_on_failure and self.is_master()):
            return
        text = self.get_message_header() + f"Here's the error:\n\n {error}\n\nTraceback:\n\n{traceback.format_exc()}"
        self.emit_message(text=text)

    def notify_on_start(self) -> None:
        """
        The notification wrapper function to be called from outside in case MML starts.

        :return:
        """
        # skip notification if either not requested or not on master node
        if not (self.do_on_start and self.is_master()):
            return
        text = "Now running:\n\n" + self.get_message_header()
        self.emit_message(text=text)

    def notify_on_end(self, return_value: Optional[float]) -> None:
        """
        The notification wrapper function to be called from outside in case MML end.

        :param Optional[float] return_value: the return value of the scheduler
        :return:
        """
        # skip notification if either not requested or not on master node
        if not (self.do_on_end and self.is_master()):
            return
        text = f"Finished (returned {return_value}):\n\n" + self.get_message_header()
        self.emit_message(text=text)

    @staticmethod
    def search_overrides() -> str:
        """
        For enriching messages the used overrides to call MML are tried to be read from the hydra config. If reading is
        not successful the returned string is a message describing the failed attempt to read

        :return: if successful the string that can be used to reproduce the MML call, otherwise a failure description
        """

        try:
            hydra_cfg = HydraConfig.get()
        except ValueError:
            hydra_cfg = None
        if hydra_cfg:
            overrides = OmegaConf.to_container(hydra_cfg.overrides)
            return str(overrides)
        else:
            # If hydra was not used, return this failure string
            return "Failed to read CLI overrides to MML! Hydra config seems to have been not set."

    @staticmethod
    def search_run_path() -> str:
        """
        For enriching messages the run path of the current experiment is tried to be read from various sources.

        :return: if successful the experiment path is extracted, otherwise a fallback notification
        """
        msg = "Run path analysis: "
        # first try to get file manager
        try:
            fm = MMLFileManager.instance()
        except TypeError:
            fm = None
        if fm:
            msg += "\n  > FileManager - " + str(fm.log_path)
            return msg
        else:
            msg += "\n  > FileManager - not instantiated"
        # second try to get hydra config
        try:
            hydra_cfg = HydraConfig.get()
        except ValueError:
            hydra_cfg = None
        if hydra_cfg:
            msg += "\n  > Hydra - " + hydra_cfg.runtime.output_dir
            return msg
        else:
            msg += "\n  > Hydra - not instantiated"
        # third pick up current working directory
        msg += "\n  > CWD - " + os.getcwd()
        return msg

    def get_message_header(self) -> str:
        """
        Helper to return uniform message header for all kinds of messages.

        :return: A header string to be used by notifiers.
        """
        return (
            f"MML version {mml.__version__}, on {socket.gethostname()}, "
            f"reporting at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Overrides: {self.search_overrides()}\n"
            f"{self.search_run_path()}\n"
        )


class EMailNotifier(BaseNotifier):
    def emit_message(self, text: str) -> None:
        """
        A function to send a simple email notification.

        All email settings are read from the mml.env environment variables (thus they have to be set beforehand):
        MML_SMTP_SERVER
        MML_SMTP_SERVER_PORT
        MML_SENDER_EMAIL
        MML_RECEIVER_EMAIL
        MML_MAIL_PASSWORD=NO_PASSWORD (set to NO_PASSWORD if there is no password necessary, otherwise set password)

        :param text: message to be sent via email
        :return:
        """
        logger.info("Sending email notification...")
        try:
            server = smtplib.SMTP(os.environ["MML_SMTP_SERVER"], port=int(os.environ["MML_SMTP_SERVER_PORT"]))
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
            if os.environ["MML_MAIL_PASSWORD"] != "NO_PASSWORD":
                server.login(os.environ["MML_SENDER_EMAIL"], os.environ["MML_MAIL_PASSWORD"])
            msg = email.message.EmailMessage()
            msg["Subject"] = "[MML] - notification"
            msg["From"] = os.environ["MML_SENDER_EMAIL"]
            msg["To"] = os.environ["MML_RECEIVER_EMAIL"]
            msg.set_content(text)
            server.send_message(msg)
            logger.info("Email notification sent!")
        except Exception as e:
            logger.error(
                "Error during sending of mail notification! Have you configured mml.env for mail notifications?"
            )
            logger.error(e)
        finally:
            server.quit()


class SlackNotifier(BaseNotifier):
    def emit_message(self, text: str) -> None:
        """
        A function to send a simple slack notification. All slack settings are read from the mml.env environment
        variables (thus they have to be set beforehand).

        A slack app has to be created via the web interface and added to the workspace. Here are some example settings:

        .. code-block:: text

            display_information:
              name: MML Monitoring
              description: Alerts from the mml monitoring system.
              background_color: "#8B0000"
            features:
              bot_user:
                display_name: MML Monitoring Alert
                always_online: true
            oauth_config:
              scopes:
                bot:
                  - incoming-webhook
            settings:
              org_deploy_enabled: false
              socket_mode_enabled: false
              token_rotation_enabled: false

        Generate the webhook and store it inside the mml.env under "MML_SLACK_WEBHOOK_URL" to enable the notifier.

        :param text: message to be sent via slack
        :return:
        """
        logger.info("Sending slack notification...")
        dump = {"username": "MML GPU Monitor", "text": text, "icon_emoji": ":information:"}
        try:
            requests.post(os.environ["MML_SLACK_WEBHOOK_URL"], json.dumps(dump))
            logger.info("Slack notification sent!")
        except Exception as e:
            logger.error(
                "Error during sending of Slack notification! Have you configured mml.env for Slack notifications?"
            )
            logger.error(e)


class DummyNotifier(BaseNotifier):
    def emit_message(self, text: str) -> None:
        """The dummy notifier does not emit any message. It may be used in testing or to avoid being messaged."""
        warnings.warn(
            "Dummy notifier does not emit any message! Please configure a different notifier to receive "
            "mml notifications."
        )
