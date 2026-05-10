import inspect
import traceback
from contextvars import copy_context
import asyncio
from functools import partial

from dash._callback_context import context_value
from dash._utils import AttributeDict
from dash.exceptions import PreventUpdate
from dash.background_callback._proxy_set_props import ProxySetProps
from dash.background_callback.managers import BaseBackgroundCallbackManager


class CeleryDiskcacheManager(BaseBackgroundCallbackManager):
    """Manage background callbacks with Celery for task execution and diskcache for result storage.

    This combines the production-grade task orchestration of Celery with
    disk-based result storage via diskcache, avoiding Redis string size limits.
    """

    def __init__(self, celery_app, cache=None, cache_by=None, expire=None):
        """
        Background callback manager that runs callback logic on a celery task queue
        and stores callback results on disk using diskcache, avoiding Redis string
        size limits.

        Celery is used for task orchestration: the broker dispatches work to workers,
        and if a result backend is configured, it is used for task state tracking
        (e.g. checking whether a job is still running). However, the actual callback
        return values, progress data, and set_props updates are stored in diskcache
        rather than the celery result backend.

        :param celery_app:
            A celery.Celery application instance. Must be configured with a broker.
            A result backend is optional but recommended — without one, task state
            checks (e.g. ``job_running``) will have limited accuracy.
        :param cache:
            A diskcache.Cache or diskcache.FanoutCache instance. If not provided,
            a diskcache.Cache instance will be created with default values.
        :param cache_by:
            A list of zero-argument functions.  When provided, caching is enabled and
            the return values of these functions are combined with the callback
            function's input arguments, triggered inputs and source code to generate
            cache keys.
        :param expire:
            If provided, a cache entry will be removed when it has not been accessed
            for ``expire`` seconds.  If not provided, the lifetime of cache entries
            is determined by the default behavior of the diskcache instance.
        """
        try:
            import celery  # type: ignore[import-not-found] # pylint: disable=import-outside-toplevel,import-error
        except ImportError as missing_imports:
            raise ImportError(
                """\
CeleryDiskcacheManager requires celery which can be installed doing

    $ pip install "dash[celery]"\n"""
            ) from missing_imports

        try:
            import diskcache  # type: ignore[import-not-found] # pylint: disable=import-outside-toplevel
        except ImportError as missing_imports:
            raise ImportError(
                """\
CeleryDiskcacheManager requires diskcache which can be installed doing

    $ pip install "dash[diskcache]"\n"""
            ) from missing_imports

        if not isinstance(celery_app, celery.Celery):
            raise ValueError("First argument must be a celery.Celery object")

        self.handle = celery_app

        if cache is None:
            self.cache = diskcache.Cache()
        else:
            if not isinstance(cache, (diskcache.Cache, diskcache.FanoutCache)):
                raise ValueError(
                    "The cache argument must be a diskcache.Cache "
                    "or diskcache.FanoutCache object"
                )
            self.cache = cache

        self.expire = expire
        super().__init__(cache_by)

    def terminate_job(self, job):
        if job is None:
            return

        self.handle.control.terminate(job)

    def terminate_unhealthy_job(self, job):
        task = self.get_task(job)
        if task and task.status in ("FAILURE", "REVOKED"):
            return self.terminate_job(job)
        return False

    def job_running(self, job):
        future = self.get_task(job)
        return future and future.status in (
            "PENDING",
            "RECEIVED",
            "STARTED",
            "RETRY",
            "PROGRESS",
        )

    def get_task(self, job):
        if job:
            return self.handle.AsyncResult(job)
        return None

    def make_job_fn(self, fn, progress, key=None):
        return _make_job_fn(fn, self.handle, self.cache, progress, key)

    def call_job_fn(self, key, job_fn, args, context):
        task = job_fn.delay(key, self._make_progress_key(key), args, context)
        return task.task_id

    def clear_cache_entry(self, key):
        self.cache.delete(key)

    def get_progress(self, key):
        progress_key = self._make_progress_key(key)
        progress_data = self.cache.get(progress_key)
        if progress_data:
            self.cache.delete(progress_key)
        return progress_data

    def result_ready(self, key):
        return self.cache.get(key) is not None

    def get_result(self, key, job):
        result = self.cache.get(key, self.UNDEFINED)
        if result is self.UNDEFINED:
            return self.UNDEFINED

        if self.cache_by is None:
            self.clear_cache_entry(key)
        else:
            if self.expire:
                self.cache.touch(key, expire=self.expire)

        self.clear_cache_entry(self._make_progress_key(key))

        self.terminate_job(job)
        return result

    def get_updated_props(self, key):
        set_props_key = self._make_set_props_key(key)
        result = self.cache.get(set_props_key, self.UNDEFINED)
        if result is self.UNDEFINED:
            return {}

        self.clear_cache_entry(set_props_key)
        return result


def _make_job_fn(fn, celery_app, cache, progress, key):  # pylint: disable=too-many-statements
    @celery_app.task(name=f"background_callback_{key}")
    def job_fn(
        result_key, progress_key, user_callback_args, context=None
    ):  # pylint: disable=too-many-statements
        def _set_progress(progress_value):
            if not isinstance(progress_value, (list, tuple)):
                progress_value = [progress_value]

            cache.set(progress_key, progress_value)

        maybe_progress = [_set_progress] if progress else []

        def _set_props(_id, props):
            cache.set(f"{result_key}-set_props", {_id: props})

        ctx = copy_context()

        def run():
            c = AttributeDict(**context)  # type: ignore[reportCallIssue]
            c.ignore_register_page = False
            c.updated_props = ProxySetProps(_set_props)
            context_value.set(c)
            errored = False
            user_callback_output = None
            try:
                if isinstance(user_callback_args, dict):
                    user_callback_output = fn(*maybe_progress, **user_callback_args)
                elif isinstance(user_callback_args, (list, tuple)):
                    user_callback_output = fn(*maybe_progress, *user_callback_args)
                else:
                    user_callback_output = fn(*maybe_progress, user_callback_args)
            except PreventUpdate:
                errored = True
                cache.set(result_key, {"_dash_no_update": "_dash_no_update"})
            except Exception as err:  # pylint: disable=broad-except
                errored = True
                cache.set(
                    result_key,
                    {
                        "background_callback_error": {
                            "msg": str(err),
                            "tb": traceback.format_exc(),
                        }
                    },
                )

            if not errored:
                cache.set(result_key, user_callback_output)

        async def async_run():
            c = AttributeDict(**context)
            c.ignore_register_page = False
            c.updated_props = ProxySetProps(_set_props)
            context_value.set(c)
            errored = False
            try:
                if isinstance(user_callback_args, dict):
                    user_callback_output = await fn(
                        *maybe_progress, **user_callback_args
                    )
                elif isinstance(user_callback_args, (list, tuple)):
                    user_callback_output = await fn(
                        *maybe_progress, *user_callback_args
                    )
                else:
                    user_callback_output = await fn(*maybe_progress, user_callback_args)
            except PreventUpdate:
                errored = True
                cache.set(result_key, {"_dash_no_update": "_dash_no_update"})
            except Exception as err:  # pylint: disable=broad-except
                errored = True
                cache.set(
                    result_key,
                    {
                        "background_callback_error": {
                            "msg": str(err),
                            "tb": traceback.format_exc(),
                        }
                    },
                )

            if asyncio.iscoroutine(user_callback_output):
                user_callback_output = await user_callback_output

            if not errored:
                cache.set(result_key, user_callback_output)

        if inspect.iscoroutinefunction(fn):
            func = partial(ctx.run, async_run)
            asyncio.run(func())
        else:
            ctx.run(run)

    return job_fn
