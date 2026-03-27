import os
from typing import Any, cast

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..logging_config import get_logger
from ..result import SWEResult
from .io import read_solutions

logger = get_logger(__name__)


def normalize_velocities_for_plotting(
    v_x: np.ndarray, v_y: np.ndarray, max_arrow_length: float, length_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize velocity vectors to limit maximum arrow length.

    Uses tanh scaling to cap large values while keeping arrows near the mean
    at reasonable sizes. All arrows scale to [0, max_arrow_length].
    """
    velocity_magnitude = np.sqrt(v_x**2 + v_y**2)
    mean = velocity_magnitude.mean()

    # Normalize by mean and apply tanh to compress outliers
    # tanh(velocity/mean) maps: 0→0, 1→0.76, 2→0.96, ∞→1
    # This caps maximum arrow length while keeping mean-sized arrows reasonable
    normalized_mag = np.tanh(velocity_magnitude / mean)
    scaled_mag = normalized_mag * max_arrow_length * length_scale

    # Scale velocity components proportionally to maintain direction
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(velocity_magnitude > 0, scaled_mag / velocity_magnitude, 0)

    v_x_scaled = v_x * scale
    v_y_scaled = v_y * scale

    return v_x_scaled, v_y_scaled


def initialize_plot(
    output_path: str,
    projection: str = "map",
    **kargs,
) -> tuple[plt.Figure, Any]:
    """Initialize a Matplotlib plot with Cartopy for Clawpack output.

    Parameters
    ----------
    output_path : str
        Path to the Clawpack output directory.
    projection : str
        Projection type: 'map' or '3d' (default: 'map').
    **kargs : dict
        Additional keyword arguments including:
        - figsize : tuple[int, int] | None
            Figure size (width, height) in inches. If None, uses default:
            (10, 8) for 3d projection, (8, 14) for map projection.
        - dark_mode : bool
            Whether to use dark background style.
        - ccrs_projection (str): for 3d only, specify ccrs projection type
            (e.g., 'platecarree', 'mollweide', 'orthographic'). If None, uses standard 3d.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axes objects.
    """
    result = SWEResult().load(os.path.join(output_path, "result.pkl"))
    X_raw, Y_raw = result.meshgrid_coord
    X = np.asarray(X_raw)
    Y = np.asarray(Y_raw)
    if X is None or Y is None:
        raise ValueError("Result file does not contain meshgrid coordinates.")

    if kargs.get("dark_mode", False):
        plt.style.use("dark_background")

    figsize = kargs.get("figsize", None)

    if projection == "3d":
        if figsize is None:
            figsize = (10, 8)
        fig = plt.figure(figsize=figsize)

        # Support ccrs projections for 3D plots
        ccrs_proj = kargs.get("ccrs_projection", None)
        if ccrs_proj:
            # Map string names to ccrs projection objects
            projection_map = {
                "platecarree": ccrs.PlateCarree(),
                "mollweide": ccrs.Mollweide(),
                "orthographic": ccrs.Orthographic(),
                "mercator": ccrs.Mercator(),
                "transversemercator": ccrs.TransverseMercator(),
            }
            ccrs_proj_obj = projection_map.get(ccrs_proj.lower(), ccrs.PlateCarree())
            ax = fig.add_subplot(1, 1, 1, projection=ccrs_proj_obj)
        else:
            ax = fig.add_subplot(1, 1, 1, projection="3d", computed_zorder=False)
    else:
        if figsize is None:
            figsize = (8, 14)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([X.min(), X.max(), Y.min(), Y.max()], crs=ccrs.PlateCarree())
        # Add satellite imagery using Google Maps tiles
        # google_tiles = cimgt.GoogleTiles(style="street")
        # ax.add_image(
        #     google_tiles,
        #     12,  # zoom level, adjust as needed
        #     interpolation="bilinear",
        # )

    return fig, ax


def plot_solution(
    output_path: str,
    frame: int = 0,
    **kargs,
) -> None:
    """Plot the solution at a given frame from Clawpack output.

    Parameters
    ----------
    output_path : str
        Path to the Clawpack output directory.
    frame : int
        Frame number to plot.
    **kargs : dict
        Additional keyword arguments including:
        - figsize : tuple[float, float] | None
            Figure size (width, height) in inches. If None, defaults to (8, 14).
    """
    figsize = kargs.get("figsize", None)

    result = read_solutions(output_path, frames_list=[frame])
    bathymetry = result["bathymetry"]
    X, Y = result["meshgrid"]
    sol = result["solutions"][0]

    h = sol[0, :, :]  # assuming depth is the first equation
    wave_treshold = kargs.get("wave_treshold", 1e-2)
    free_surface = bathymetry + h
    free_surface[h < wave_treshold] = np.nan
    v_x, v_y = sol[1, :, :] / h, sol[2, :, :] / h  # velocities

    v_x[h <= wave_treshold] = 0.0
    v_y[h <= wave_treshold] = 0.0

    max_arrow_length = kargs.get("max_arrow_length", 0.5)
    length_scale = kargs.get("length_scale", 1.0)
    v_x_scaled, v_y_scaled = normalize_velocities_for_plotting(
        v_x, v_y, max_arrow_length, length_scale
    )

    velocity = np.sqrt(v_x**2 + v_y**2)
    plt.style.use("dark_background")
    h_fig = plt.figure(figsize=figsize)
    # Add a satellital image using cartopy's Stamen imagery
    h_ax = h_fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    h_ax.set_extent([X.min(), X.max(), Y.min(), Y.max()], crs=ccrs.PlateCarree())
    # Add satellite imagery using Google Maps tiles

    google_tiles = cimgt.GoogleTiles(style="satellite")
    h_ax.add_image(
        google_tiles,
        17,  # zoom level, adjust as needed
        interpolation="bilinear",
    )

    contour = h_ax.contourf(
        X, Y, bathymetry, levels=50, cmap="viridis", alpha=0.7
    )  # perceptually uniform, good for scalar fields
    arrow_step = kargs.get("arrow_step", 2)  # show every nth arrow
    velocity_field = h_ax.quiver(
        X[::arrow_step, ::arrow_step],
        Y[::arrow_step, ::arrow_step],
        v_x_scaled[::arrow_step, ::arrow_step],
        v_y_scaled[::arrow_step, ::arrow_step],
        velocity[::arrow_step, ::arrow_step],
        angles="xy",
        cmap="cool",  # sequential, high contrast, visually distinct from 'viridis'
        width=0.005,  # make arrows wider (default is ~0.002)
        scale=1 / 0.1,  # adjust as needed for arrow length scaling
    )
    divider = make_axes_locatable(h_ax)
    cax1 = divider.new_horizontal(size="5%", pad=0.05, axes_class=plt.Axes)
    cax2 = divider.new_horizontal(size="5%", pad=0.75, axes_class=plt.Axes)
    h_fig.colorbar(contour, cax=cax1, label="Water Level (m)")
    h_fig.colorbar(velocity_field, cax=cax2, label="Velocity Magnitude (m/s)")
    h_fig.add_axes(cax1)
    h_fig.add_axes(cax2)
    h_ax.set_xlabel("Longitude")
    h_ax.set_ylabel("Latitude")
    h_ax.set_title(f"Wave height at time {result['times'][frame]}")
    h_ax.set_xticks(np.linspace(X.min(), X.max(), 6))
    h_ax.set_yticks(np.linspace(Y.min(), Y.max(), 6))
    # if kargs.get("dark", True):
    h_fig.tight_layout()
    h_fig.savefig(
        os.path.join(output_path, f"solution_time_{result['times'][frame]}.png"),
        bbox_inches="tight",
        transparent=True,
        dpi=200,
    )

    plt.show()


def animate_solution(
    output_path: str,
    frames: list[int] | None = None,
    **kargs,
) -> None:
    """Create an animation of solutions over multiple frames from Clawpack output.

    Parameters
    ----------
    output_path : str
        Path to the Clawpack output directory.
    frames : list[int] | None
        List of frame numbers to animate, or None for all frames.
    **kargs : dict
        Additional keyword arguments for plotting.
        Supported options include:
        - figsize : tuple[float, float] | None
            Figure size (width, height) in inches. If None, defaults to (8, 14).
        - dark_mode (bool): if True, use a dark plot background.
        - mpl_rc_params (dict): dictionary of matplotlib rcParams to update before plotting.
    """
    import matplotlib as mpl
    import matplotlib.animation as animation

    # Apply custom matplotlib rcParams if provided
    mpl_rc_params = kargs.get("mpl_rc_params", {})
    if mpl_rc_params:
        mpl.rcParams.update(mpl_rc_params)

    result = read_solutions(output_path, frames_list=frames)
    bathymetry = result["bathymetry"]
    X, Y = result["meshgrid"]
    solutions = result["solutions"]
    fig, ax = initialize_plot(output_path, **kargs)
    ax = cast(Any, ax)

    wave_treshold = kargs.get("wave_treshold", 1e-2)
    max_arrow_length = kargs.get("max_arrow_length", 0.5)

    contourf: list[Any | None] = [None]
    quiver: list[Any | None] = [None]
    colorbar1: list[Any | None] = [None]
    colorbar2: list[Any | None] = [None]

    divider = make_axes_locatable(ax)
    cax1 = divider.new_horizontal(size="5%", pad=0.05, axes_class=plt.Axes)
    cax2 = divider.new_horizontal(size="5%", pad=0.75, axes_class=plt.Axes)
    fig.add_axes(cax1)
    fig.add_axes(cax2)

    def update(frame_idx: int) -> list[Any]:
        nonlocal cax1, cax2, ax

        ax.clear()
        cax1.clear()
        cax2.clear()
        # colorbar1.remove()
        # colorbar2.remove()
        # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([X.min(), X.max(), Y.min(), Y.max()], crs=ccrs.PlateCarree())
        # google_tiles = cimgt.GoogleTiles(style="street")
        # ax.add_image(google_tiles, 14, interpolation="bilinear")

        sol = solutions[frame_idx]
        h = sol[0, :, :]
        free_surface = bathymetry + h
        mask = h < wave_treshold
        free_surface[mask] = np.nan
        h_div = np.where(h <= wave_treshold, 1.0, h)
        v_x, v_y = sol[1, :, :] / h_div, sol[2, :, :] / h_div
        v_x[h <= wave_treshold] = 0.0
        v_y[h <= wave_treshold] = 0.0
        length_scale = kargs.get("length_scale", 1.0)
        v_x_scaled, v_y_scaled = normalize_velocities_for_plotting(
            v_x, v_y, max_arrow_length, length_scale
        )
        velocity = np.sqrt(v_x**2 + v_y**2)

        contourf[0] = ax.contourf(
            X, Y, free_surface, levels=30, cmap="viridis", alpha=0.7
        )
        v_x_scaled[mask] = np.nan
        v_y_scaled[mask] = np.nan
        arrow_step = kargs.get("arrow_step", 2)  # show every nth arrow
        quiver[0] = ax.quiver(
            X[::arrow_step, ::arrow_step],
            Y[::arrow_step, ::arrow_step],
            v_x_scaled[::arrow_step, ::arrow_step],
            v_y_scaled[::arrow_step, ::arrow_step],
            velocity[::arrow_step, ::arrow_step],
            angles="xy",
            cmap="cool",
            width=0.005,
            scale=1 / 0.1,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Wave height at time {result['times'][frame_idx]:.2f} s")
        ax.set_xticks(np.linspace(X.min(), X.max(), 4))
        ax.set_yticks(np.linspace(Y.min(), Y.max(), 6))
        # if dark_mode:
        #     ax.set_facecolor("black")
        # else:
        #     ax.set_facecolor("white")
        # ax.xaxis.label.set_color("white")
        # ax.yaxis.label.set_color("white")
        # ax.title.set_color("white")
        # ax.tick_params(axis="x", colors="white")
        # ax.tick_params(axis="y", colors="white")
        # Update colorbars for the new contour and quiver plots
        assert contourf[0] is not None and quiver[0] is not None

        colorbar1[0] = fig.colorbar(
            contourf[0],
            ax=ax,
            cax=cax1,
            label="Water Level (m)",
        )
        colorbar2[0] = fig.colorbar(
            quiver[0],
            ax=ax,
            cax=cax2,
            label="Velocity Magnitude (m/s)",
        )
        return [contourf[0], quiver[0]]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=solutions.shape[0],
        interval=int(kargs.get("interval", 200)),
    )
    if kargs.get("save", False):
        ani.save(
            os.path.join(output_path, kargs.get("file_name", "wave_animation.mp4")),
            writer=kargs.get("writer", "ffmpeg"),
            dpi=200,
            fps=int(kargs.get("fps", 50)),
        )
    else:
        plt.show()

    return


def animate_surface(
    output_path: str,
    frames: list[int] | None = None,
    **kargs,
) -> None:
    """Create a 3D surface animation of wave solutions from Clawpack output.

    Parameters
    ----------
    output_path : str
        Path to the Clawpack output directory.
    frames : list[int] | None
        List of frame indices to animate, or None for all frames.
    **kargs : dict
        Additional keyword arguments for plotting/export.
        Supported options include:
        - figsize : tuple[float, float] | None
            Figure size (width, height) in inches. If None, defaults to (10, 8).
        - wave_treshold (float): mask threshold for wet cells (default: 1e-3)
        - interval (int): animation interval in ms (default: 50)
        - elev (float): camera elevation in degrees (default: 30.0)
        - azim (float): camera azimuth in degrees (default: -120.0)
        - dark_mode (bool): if True, use dark style/background
        - save (bool): if True, save animation instead of showing it
        - file_name (str): output filename (default: wave_surface_animation.mp4)
        - writer (str): animation writer backend (default: ffmpeg)
        - fps (int): output frame rate when saving (default: 30)
        - mpl_rc_params (dict): dictionary of matplotlib rcParams to update before plotting.
        - ccrs_projection (str): ccrs projection type for 3D axes
            (e.g., 'platecarree', 'mollweide', 'orthographic'). If None, uses standard 3d.
    """
    import matplotlib as mpl
    import matplotlib.animation as animation

    # Apply custom matplotlib rcParams if provided
    mpl_rc_params = kargs.get("mpl_rc_params", {})
    if mpl_rc_params:
        mpl.rcParams.update(mpl_rc_params)

    result = read_solutions(output_path, frames_list=frames)
    bathymetry = result["bathymetry"]
    X, Y = result["meshgrid"]
    solutions = result["solutions"]
    wave_treshold = float(kargs.get("wave_treshold", 1e-3))
    interval = int(kargs.get("interval", 50))
    elev = float(kargs.get("elev", 30.0))
    azim = float(kargs.get("azim", -120.0))
    dark_mode = bool(kargs.get("dark_mode", False))

    fig, ax = initialize_plot(
        output_path,
        projection="3d",
        **kargs,
    )
    ax = cast(Any, ax)
    cax = fig.add_axes((0.8, 0.16, 0.03, 0.68))
    fig.subplots_adjust(right=0.86)

    x_min, x_max = float(np.nanmin(X)), float(np.nanmax(X))
    y_min, y_max = float(np.nanmin(Y)), float(np.nanmax(Y))
    z_min = float(np.nanmin(bathymetry))
    z_max = float(np.nanmax(bathymetry + solutions[:, 0, :, :]))

    colorbar: Any | None = None

    def update(frame_idx: int) -> tuple[Any]:
        nonlocal colorbar
        ax.clear()
        cax.clear()

        sol = solutions[frame_idx]
        h = sol[0, :, :]
        free_surface = bathymetry + h
        free_surface[h < wave_treshold] = np.nan

        # Plot bathymetry first (underneath with lower zorder)
        ax.plot_surface(
            X,
            Y,
            bathymetry,
            cmap="BrBG",
            linewidth=0,
            antialiased=False,
            alpha=0.7,
            zorder=1,
        )

        # Plot water surface on top (higher zorder)
        surface = ax.plot_surface(
            X,
            Y,
            free_surface,
            cmap="viridis",
            linewidth=0,
            antialiased=False,
            alpha=0.95,
            vmin=float(np.nanmin(free_surface)),
            vmax=float(np.nanmax(free_surface)),
            zorder=2,
        )
        colorbar = fig.colorbar(surface, cax=cax, label="Surface Elevation (m)")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min - 0.1, z_max + 0.1)
        ax.set_zlabel("Surface Elevation (m)")
        ax.set_title(f"Wave surface at time {result['times'][frame_idx]:.2f} s")
        ax.view_init(elev=elev, azim=azim)
        ax.set_xticks([])
        ax.set_yticks([])

        if dark_mode:
            ax.set_facecolor("black")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.zaxis.label.set_color("white")
            ax.title.set_color("white")
            ax.tick_params(axis="x", colors="white")
            ax.tick_params(axis="y", colors="white")
            ax.tick_params(axis="z", colors="white")

        return (surface,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=solutions.shape[0],
        interval=interval,
    )

    if kargs.get("save", False):
        ani.save(
            os.path.join(
                output_path,
                str(kargs.get("file_name", "wave_surface_animation.mp4")),
            ),
            writer=str(kargs.get("writer", "ffmpeg")),
            dpi=200,
            fps=int(kargs.get("fps", 30)),
        )
    else:
        plt.show()


if __name__ == "__main__":
    # plot_solution(output_path="_output", frame=20, wave_treshold=1e-2)
    animate_solution(
        output_path="../../_output",
        frames=list(range(0, 2000)),
        wave_treshold=1e-2,
        interval=100,
        save=False,
    )
