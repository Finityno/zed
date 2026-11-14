#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

// `SpriteEffect::kind`: `0` is plain text, these two are the other cases.
// Metal applies no contrast or gamma ramp to a monochrome sprite, so here a
// mask tile differs from plain text only in having no shimmer band.
constant uint SPRITE_EFFECT_SHIMMER = 1u;

// Set in a flat instance id when the fragment shader can shade the primitive
// from its interpolated inputs alone, without loading the instance.
constant uint SIMPLE_PRIMITIVE_FLAG = 1u << 31;

float4 hsla_to_rgba(Hsla hsla);
float3 srgb_to_linear(float3 color);
float3 linear_to_srgb(float3 color);
float4 srgb_to_oklab(float4 color);
float4 oklab_to_srgb(float4 color);
float4 to_device_position_impl(float2 position,
                               constant Size_DevicePixels *viewport_size);
float4 to_device_position(float2 unit_vertex, Bounds_ScaledPixels bounds,
                          constant Size_DevicePixels *viewport_size);
float4 to_device_position_transformed(float2 unit_vertex, Bounds_ScaledPixels bounds,
                          TransformationMatrix transformation,
                          constant Size_DevicePixels *input_viewport_size);
float quad_depth(uint quad_id);
float rounded_clip_coverage(uint index, float2 position, constant RoundedClipNode *masks);
float rounded_clip_interior_inset(RoundedClipNode mask);

float2 to_tile_position(float2 unit_vertex, AtlasTile tile,
                        constant Size_DevicePixels *atlas_size);
float4 distance_from_clip_rect_transformed(float2 unit_vertex, Bounds_ScaledPixels bounds,
                               Bounds_ScaledPixels clip_bounds, TransformationMatrix transformation);

struct ClippedVertex {
  float2 position;
  // The corner as a fraction of the original bounds.
  float2 unit_vertex;
};

ClippedVertex clip_to_mask(float2 unit_vertex, Bounds_ScaledPixels bounds,
                           Bounds_ScaledPixels mask);
float corner_dash_velocity(float dv1, float dv2);
float dash_alpha(float t, float period, float length, float dash_velocity,
                 float antialias_threshold);
float quarter_ellipse_sdf(float2 point, float2 radii);
float pick_corner_radius(float2 center_to_point, Corners_ScaledPixels corner_radii);
float quad_sdf(float2 point, Bounds_ScaledPixels bounds,
               Corners_ScaledPixels corner_radii);
float quad_sdf_impl(float2 center_to_point, float corner_radius);
float4 over(float4 below, float4 above);
float radians(float degrees);
float4 fill_color(Background background, float2 position, Bounds_ScaledPixels bounds,
  float4 solid_color, float4 color0, float4 color1);
float4 quad_solid_interior(Quad quad, constant RoundedClipNode *masks);
bool coordinates_are_precise(Bounds_ScaledPixels bounds);

struct GradientColor {
  float4 solid;
  float4 color0;
  float4 color1;
};
GradientColor prepare_fill_color(uint tag, uint color_space, Hsla solid, Hsla color0, Hsla color1);

struct QuadVertexOutput {
  uint rounded_clip [[flat]];
  uint quad_id [[flat]];
  float4 position [[position]];
  float4 border_color [[flat]];
  float4 background_solid [[flat]];
  float4 background_color0 [[flat]];
  float4 background_color1 [[flat]];
  float4 solid_interior [[flat]];
};

struct QuadFragmentInput {
  uint rounded_clip [[flat]];
  uint quad_id [[flat]];
  float4 position [[position]];
  float4 border_color [[flat]];
  float4 background_solid [[flat]];
  float4 background_color0 [[flat]];
  float4 background_color1 [[flat]];
  float4 solid_interior [[flat]];
};

vertex QuadVertexOutput quad_vertex(uint unit_vertex_id [[vertex_id]],
                                    uint instance_id [[instance_id]],
                                    constant float2 *unit_vertices
                                    [[buffer(QuadInputIndex_Vertices)]],
                                    constant Quad *quads
                                    [[buffer(QuadInputIndex_Quads)]],
                                    constant uint *quad_indices
                                    [[buffer(QuadInputIndex_QuadIndices)]],
                                    constant Size_DevicePixels *viewport_size
                                    [[buffer(QuadInputIndex_ViewportSize)]],
                                    constant RoundedClipNode *rounded_masks [[buffer(15)]]) {
  float2 unit_vertex = unit_vertices[unit_vertex_id];
  uint quad_id = quad_indices[instance_id];
  Quad quad = quads[quad_id];
  ClippedVertex clipped =
      clip_to_mask(unit_vertex, quad.bounds, quad.content_mask.bounds);
  float4 device_position =
      to_device_position_impl(clipped.position, viewport_size);
  device_position.z = quad_depth(quad_id);
  float4 border_color = hsla_to_rgba(quad.border_color);

  GradientColor gradient = prepare_fill_color(
    quad.background.tag,
    quad.background.color_space,
    quad.background.solid,
    quad.background.colors[0].color,
    quad.background.colors[1].color
  );

  return QuadVertexOutput{
      quad.rounded_clip,
      quad_id,
      device_position,
      border_color,
      gradient.solid,
      gradient.color0,
      gradient.color1,
      quad_solid_interior(quad, rounded_masks)};
}

// Whether `bounds` lies where a float resolves well under the margins the
// shortcuts below keep: below 2^18 a coordinate's step is 1/32 of a pixel,
// so computing an edge directly and computing it through a centre and half
// size, as the distance fields do, cannot disagree across them.
bool coordinates_are_precise(Bounds_ScaledPixels bounds) {
  const float limit = 262144.0;
  float reach = max(max(fabs(bounds.origin.x), fabs(bounds.origin.y)),
                    max(fabs(bounds.origin.x + bounds.size.width),
                        fabs(bounds.origin.y + bounds.size.height)));
  return reach < limit;
}

// The rectangle, as (left, top, right, bottom) in device pixels, in which
// every pixel of a solid-filled quad is its plain background at full clip
// coverage: farther from each edge than any corner radius and any border
// reaches, with a pixel to spare for antialiasing (an unrounded, borderless
// quad has neither, so all of it), and inside every rounded clip it draws in
// by more than that clip's corners and antialiasing reach. Empty for every
// other fill, whose colour depends on the position. The fragment shader
// returns the background there without loading the quad, evaluating its
// outline or walking its clips, which for a large translucent panel is
// nearly every pixel it covers.
float4 quad_solid_interior(Quad quad, constant RoundedClipNode *masks) {
  const float4 empty = float4(1.0, 1.0, 0.0, 0.0);
  if (quad.background.tag != 0 || !coordinates_are_precise(quad.bounds)) {
    return empty;
  }
  float radius = max(max(quad.corner_radii.top_left, quad.corner_radii.top_right),
                     max(quad.corner_radii.bottom_right, quad.corner_radii.bottom_left));
  float border = max(max(quad.border_widths.top, quad.border_widths.right),
                     max(quad.border_widths.bottom, quad.border_widths.left));
  float inset = (radius == 0.0 && border == 0.0) ? 0.0 : max(radius, border) + 1.0;
  float4 interior = float4(quad.bounds.origin.x + inset,
                           quad.bounds.origin.y + inset,
                           quad.bounds.origin.x + quad.bounds.size.width - inset,
                           quad.bounds.origin.y + quad.bounds.size.height - inset);
  uint clip = quad.rounded_clip;
  while (clip != 0) {
    RoundedClipNode mask = masks[clip - 1];
    if (!coordinates_are_precise(mask.bounds)) {
      return empty;
    }
    float mask_inset = rounded_clip_interior_inset(mask);
    interior = float4(max(interior.x, mask.bounds.origin.x + mask_inset),
                      max(interior.y, mask.bounds.origin.y + mask_inset),
                      min(interior.z, mask.bounds.origin.x + mask.bounds.size.width - mask_inset),
                      min(interior.w, mask.bounds.origin.y + mask.bounds.size.height - mask_inset));
    clip = mask.parent;
  }
  return interior;
}

fragment float4 quad_fragment(QuadFragmentInput input [[stage_in]],
                              constant Quad *quads
                              [[buffer(QuadInputIndex_Quads)]],
    constant RoundedClipNode *rounded_masks [[buffer(15)]]) {
  float2 position = input.position.xy;
  if (position.x > input.solid_interior.x && position.y > input.solid_interior.y &&
      position.x < input.solid_interior.z && position.y < input.solid_interior.w) {
    return input.background_solid;
  }
  Quad quad = quads[input.quad_id];
  float4 background_color = fill_color(quad.background, input.position.xy, quad.bounds,
    input.background_solid, input.background_color0, input.background_color1);

  bool unrounded = quad.corner_radii.top_left == 0.0 &&
    quad.corner_radii.bottom_left == 0.0 &&
    quad.corner_radii.top_right == 0.0 &&
    quad.corner_radii.bottom_right == 0.0;

  // Fast path when the quad is not rounded and doesn't have any border
  if (quad.border_widths.top == 0.0 &&
      quad.border_widths.left == 0.0 &&
      quad.border_widths.right == 0.0 &&
      quad.border_widths.bottom == 0.0 &&
      unrounded) {
    return (background_color) * float4(1.0, 1.0, 1.0, rounded_clip_coverage(input.rounded_clip, input.position.xy, rounded_masks));
  }

  float2 size = float2(quad.bounds.size.width, quad.bounds.size.height);
  float2 half_size = size / 2.0;
  float2 point = input.position.xy - float2(quad.bounds.origin.x, quad.bounds.origin.y);
  float2 center_to_point = point - half_size;

  // Signed distance field threshold for inclusion of pixels. 0.5 is the
  // minimum distance between the center of the pixel and the edge.
  const float antialias_threshold = 0.5;

  // Radius of the nearest corner
  float corner_radius = pick_corner_radius(center_to_point, quad.corner_radii);

  // Width of the nearest borders
  float2 border = float2(
    center_to_point.x < 0.0 ? quad.border_widths.left : quad.border_widths.right,
    center_to_point.y < 0.0 ? quad.border_widths.top : quad.border_widths.bottom
  );

  // 0-width borders are reduced so that `inner_sdf >= antialias_threshold`.
  // The purpose of this is to not draw antialiasing pixels in this case.
  float2 reduced_border = float2(
    border.x == 0.0 ? -antialias_threshold : border.x,
    border.y == 0.0 ? -antialias_threshold : border.y);

  // Vector from the corner of the quad bounds to the point, after mirroring
  // the point into the bottom right quadrant. Both components are <= 0.
  float2 corner_to_point = fabs(center_to_point) - half_size;

  // Vector from the point to the center of the rounded corner's circle, also
  // mirrored into bottom right quadrant.
  float2 corner_center_to_point = corner_to_point + corner_radius;

  // Whether the nearest point on the border is rounded
  bool is_near_rounded_corner =
    corner_center_to_point.x >= 0.0 &&
    corner_center_to_point.y >= 0.0;

  // Vector from straight border inner corner to point.
  //
  // 0-width borders are turned into width -1 so that inner_sdf is > 1.0 near
  // the border. Without this, antialiasing pixels would be drawn.
  float2 straight_border_inner_corner_to_point = corner_to_point + reduced_border;

  // Whether the point is beyond the inner edge of the straight border
  bool is_beyond_inner_straight_border =
    straight_border_inner_corner_to_point.x > 0.0 ||
    straight_border_inner_corner_to_point.y > 0.0;


  // Whether the point is far enough inside the quad, such that the pixels are
  // not affected by the straight border.
  bool is_within_inner_straight_border =
    straight_border_inner_corner_to_point.x < -antialias_threshold &&
    straight_border_inner_corner_to_point.y < -antialias_threshold;

  // Fast path for points that must be part of the background
  if (is_within_inner_straight_border && !is_near_rounded_corner) {
    return (background_color) * float4(1.0, 1.0, 1.0, rounded_clip_coverage(input.rounded_clip, input.position.xy, rounded_masks));
  }

  // Signed distance of the point to the outside edge of the quad's border
  float outer_sdf = quad_sdf_impl(corner_center_to_point, corner_radius);

  // Approximate signed distance of the point to the inside edge of the quad's
  // border. It is negative outside this edge (within the border), and
  // positive inside.
  //
  // This is not always an accurate signed distance:
  // * The rounded portions with varying border width use an approximation of
  //   nearest-point-on-ellipse.
  // * When it is quickly known to be outside the edge, -1.0 is used.
  float inner_sdf = 0.0;
  if (corner_center_to_point.x <= 0.0 || corner_center_to_point.y <= 0.0) {
    // Fast paths for straight borders
    inner_sdf = -max(straight_border_inner_corner_to_point.x,
                     straight_border_inner_corner_to_point.y);
  } else if (is_beyond_inner_straight_border) {
    // Fast path for points that must be outside the inner edge
    inner_sdf = -1.0;
  } else if (reduced_border.x == reduced_border.y) {
    // Fast path for circular inner edge.
    inner_sdf = -(outer_sdf + reduced_border.x);
  } else {
    float2 ellipse_radii = max(float2(0.0), float2(corner_radius) - reduced_border);
    inner_sdf = quarter_ellipse_sdf(corner_center_to_point, ellipse_radii);
  }

  // Negative when inside the border
  float border_sdf = max(inner_sdf, outer_sdf);

  float4 color = background_color;
  if (border_sdf < antialias_threshold) {
    float4 border_color = input.border_color;

    // Dashed border logic when border_style == 1
    if (quad.border_style == 1) {
      // Position along the perimeter in "dash space", where each dash
      // period has length 1
      float t = 0.0;

      // Total number of dash periods, so that the dash spacing can be
      // adjusted to evenly divide it
      float max_t = 0.0;

      // Border width is proportional to dash size. This is the behavior
      // used by browsers, but also avoids dashes from different segments
      // overlapping when dash size is smaller than the border width.
      //
      // Dash pattern: (2 * border width) dash, (1 * border width) gap
      const float dash_length_per_width = 2.0;
      const float dash_gap_per_width = 1.0;
      const float dash_period_per_width = dash_length_per_width + dash_gap_per_width;

      // Since the dash size is determined by border width, the density of
      // dashes varies. Multiplying a pixel distance by this returns a
      // position in dash space - it has units (dash period / pixels). So
      // a dash velocity of (1 / 10) is 1 dash every 10 pixels.
      float dash_velocity = 0.0;

      // Dividing this by the border width gives the dash velocity
      const float dv_numerator = 1.0 / dash_period_per_width;

      if (unrounded) {
        // When corners aren't rounded, the dashes are separately laid
        // out on each straight line, rather than around the whole
        // perimeter. This way each line starts and ends with a dash.
        bool is_horizontal = corner_center_to_point.x < corner_center_to_point.y;

        // Choosing the right border width for dashed borders.
        // TODO: A better solution exists taking a look at the whole file.
        // this does not fix single dashed borders at the corners
        float2 dashed_border = float2(
        fmax(quad.border_widths.bottom, quad.border_widths.top),
        fmax(quad.border_widths.right, quad.border_widths.left));

        float border_width = is_horizontal ? dashed_border.x : dashed_border.y;
        dash_velocity = dv_numerator / border_width;
        t = is_horizontal ? point.x : point.y;
        t *= dash_velocity;
        max_t = is_horizontal ? size.x : size.y;
        max_t *= dash_velocity;
      } else {
        // When corners are rounded, the dashes are laid out clockwise
        // around the whole perimeter.

        float r_tr = quad.corner_radii.top_right;
        float r_br = quad.corner_radii.bottom_right;
        float r_bl = quad.corner_radii.bottom_left;
        float r_tl = quad.corner_radii.top_left;

        float w_t = quad.border_widths.top;
        float w_r = quad.border_widths.right;
        float w_b = quad.border_widths.bottom;
        float w_l = quad.border_widths.left;

        // Straight side dash velocities
        float dv_t = w_t <= 0.0 ? 0.0 : dv_numerator / w_t;
        float dv_r = w_r <= 0.0 ? 0.0 : dv_numerator / w_r;
        float dv_b = w_b <= 0.0 ? 0.0 : dv_numerator / w_b;
        float dv_l = w_l <= 0.0 ? 0.0 : dv_numerator / w_l;

        // Straight side lengths in dash space
        float s_t = (size.x - r_tl - r_tr) * dv_t;
        float s_r = (size.y - r_tr - r_br) * dv_r;
        float s_b = (size.x - r_br - r_bl) * dv_b;
        float s_l = (size.y - r_bl - r_tl) * dv_l;

        float corner_dash_velocity_tr = corner_dash_velocity(dv_t, dv_r);
        float corner_dash_velocity_br = corner_dash_velocity(dv_b, dv_r);
        float corner_dash_velocity_bl = corner_dash_velocity(dv_b, dv_l);
        float corner_dash_velocity_tl = corner_dash_velocity(dv_t, dv_l);

        // Corner lengths in dash space
        float c_tr = r_tr * (M_PI_F / 2.0) * corner_dash_velocity_tr;
        float c_br = r_br * (M_PI_F / 2.0) * corner_dash_velocity_br;
        float c_bl = r_bl * (M_PI_F / 2.0) * corner_dash_velocity_bl;
        float c_tl = r_tl * (M_PI_F / 2.0) * corner_dash_velocity_tl;

        // Cumulative dash space upto each segment
        float upto_tr = s_t;
        float upto_r = upto_tr + c_tr;
        float upto_br = upto_r + s_r;
        float upto_b = upto_br + c_br;
        float upto_bl = upto_b + s_b;
        float upto_l = upto_bl + c_bl;
        float upto_tl = upto_l + s_l;
        max_t = upto_tl + c_tl;

        if (is_near_rounded_corner) {
          float radians = atan2(corner_center_to_point.y, corner_center_to_point.x);
          float corner_t = radians * corner_radius;

          if (center_to_point.x >= 0.0) {
            if (center_to_point.y < 0.0) {
              dash_velocity = corner_dash_velocity_tr;
              // Subtracted because radians is pi/2 to 0 when
              // going clockwise around the top right corner,
              // since the y axis has been flipped
              t = upto_r - corner_t * dash_velocity;
            } else {
              dash_velocity = corner_dash_velocity_br;
              // Added because radians is 0 to pi/2 when going
              // clockwise around the bottom-right corner
              t = upto_br + corner_t * dash_velocity;
            }
          } else {
            if (center_to_point.y >= 0.0) {
              dash_velocity = corner_dash_velocity_bl;
              // Subtracted because radians is pi/1 to 0 when
              // going clockwise around the bottom-left corner,
              // since the x axis has been flipped
              t = upto_l - corner_t * dash_velocity;
            } else {
              dash_velocity = corner_dash_velocity_tl;
              // Added because radians is 0 to pi/2 when going
              // clockwise around the top-left corner, since both
              // axis were flipped
              t = upto_tl + corner_t * dash_velocity;
            }
          }
        } else {
          // Straight borders
          bool is_horizontal = corner_center_to_point.x < corner_center_to_point.y;
          if (is_horizontal) {
            if (center_to_point.y < 0.0) {
              dash_velocity = dv_t;
              t = (point.x - r_tl) * dash_velocity;
            } else {
              dash_velocity = dv_b;
              t = upto_bl - (point.x - r_bl) * dash_velocity;
            }
          } else {
            if (center_to_point.x < 0.0) {
              dash_velocity = dv_l;
              t = upto_tl - (point.y - r_tl) * dash_velocity;
            } else {
              dash_velocity = dv_r;
              t = upto_r + (point.y - r_tr) * dash_velocity;
            }
          }
        }
      }

      float dash_length = dash_length_per_width / dash_period_per_width;
      float desired_dash_gap = dash_gap_per_width / dash_period_per_width;

      // Straight borders should start and end with a dash, so max_t is
      // reduced to cause this.
      max_t -= unrounded ? dash_length : 0.0;
      if (max_t >= 1.0) {
        // Adjust dash gap to evenly divide max_t
        float dash_count = floor(max_t);
        float dash_period = max_t / dash_count;
        border_color.a *= dash_alpha(t, dash_period, dash_length, dash_velocity,
                                     antialias_threshold);
      } else if (unrounded) {
        // When there isn't enough space for the full gap between the
        // two start / end dashes of a straight border, reduce gap to
        // make them fit.
        float dash_gap = max_t - dash_length;
        if (dash_gap > 0.0) {
          float dash_period = dash_length + dash_gap;
          border_color.a *= dash_alpha(t, dash_period, dash_length, dash_velocity,
                                       antialias_threshold);
        }
      }
    }

    // Blend the border on top of the background and then linearly interpolate
    // between the two as we slide inside the background.
    float4 blended_border = over(background_color, border_color);
    color = mix(background_color, blended_border,
                saturate(antialias_threshold - inner_sdf));
  }

  return (color * float4(1.0, 1.0, 1.0, saturate(antialias_threshold - outer_sdf))) * float4(1.0, 1.0, 1.0, rounded_clip_coverage(input.rounded_clip, input.position.xy, rounded_masks));
}

// Returns the dash velocity of a corner given the dash velocity of the two
// sides, by returning the slower velocity (larger dashes).
//
// Since 0 is used for dash velocity when the border width is 0 (instead of
// +inf), this returns the other dash velocity in that case.
//
// An alternative to this might be to appropriately interpolate the dash
// velocity around the corner, but that seems overcomplicated.
float corner_dash_velocity(float dv1, float dv2) {
  if (dv1 == 0.0) {
    return dv2;
  } else if (dv2 == 0.0) {
    return dv1;
  } else {
    return min(dv1, dv2);
  }
}

// Returns alpha used to render antialiased dashes.
// `t` is within the dash when `fmod(t, period) < length`.
float dash_alpha(
    float t, float period, float length, float dash_velocity,
    float antialias_threshold) {
  float half_period = period / 2.0;
  float half_length = length / 2.0;
  // Value in [-half_period, half_period]
  // The dash is in [-half_length, half_length]
  float centered = fmod(t + half_period - half_length, period) - half_period;
  // Signed distance for the dash, negative values are inside the dash
  float signed_distance = abs(centered) - half_length;
  // Antialiased alpha based on the signed distance
  return saturate(antialias_threshold - signed_distance / dash_velocity);
}

// This approximates distance to the nearest point to a quarter ellipse in a way
// that is sufficient for anti-aliasing when the ellipse is not very eccentric.
// The components of `point` are expected to be positive.
//
// Negative on the outside and positive on the inside.
float quarter_ellipse_sdf(float2 point, float2 radii) {
  // Scale the space to treat the ellipse like a unit circle
  float2 circle_vec = point / radii;
  float unit_circle_sdf = length(circle_vec) - 1.0;
  // Approximate up-scaling of the length by using the average of the radii.
  //
  // TODO: A better solution would be to use the gradient of the implicit
  // function for an ellipse to approximate a scaling factor.
  return unit_circle_sdf * (radii.x + radii.y) * -0.5;
}

struct OpaqueQuadVertexOutput {
  float4 position [[position]];
  float4 color [[flat]];
};

struct OpaqueQuadFragmentInput {
  float4 position [[position]];
  float4 color [[flat]];
};

Bounds_ScaledPixels opaque_quad_core(Quad quad) {
  bool has_rounded_corners = quad.corner_radii.top_left != 0.0 ||
                             quad.corner_radii.top_right != 0.0 ||
                             quad.corner_radii.bottom_right != 0.0 ||
                             quad.corner_radii.bottom_left != 0.0;
  if (!has_rounded_corners) {
    return quad.bounds;
  }

  const float antialias_inset = 1.0;
  float left_inset =
      max(quad.corner_radii.top_left, quad.corner_radii.bottom_left) +
      antialias_inset;
  float right_inset =
      max(quad.corner_radii.top_right, quad.corner_radii.bottom_right) +
      antialias_inset;
  float top_inset =
      max(quad.corner_radii.top_left, quad.corner_radii.top_right) +
      antialias_inset;
  float bottom_inset =
      max(quad.corner_radii.bottom_left, quad.corner_radii.bottom_right) +
      antialias_inset;

  float2 size = float2(quad.bounds.size.width, quad.bounds.size.height);

  // A horizontal band avoids the top and bottom corner arcs; a vertical band
  // avoids the left and right arcs. Either is fully opaque, so use the larger.
  float2 horizontal_band = max(
      size - float2(2.0 * antialias_inset, top_inset + bottom_inset),
      float2(0.0));
  float2 vertical_band = max(
      size - float2(left_inset + right_inset, 2.0 * antialias_inset),
      float2(0.0));

  Bounds_ScaledPixels core = quad.bounds;
  if (horizontal_band.x * horizontal_band.y >=
      vertical_band.x * vertical_band.y) {
    core.origin.x += antialias_inset;
    core.origin.y += top_inset;
    core.size.width = horizontal_band.x;
    core.size.height = horizontal_band.y;
  } else {
    core.origin.x += left_inset;
    core.origin.y += antialias_inset;
    core.size.width = vertical_band.x;
    core.size.height = vertical_band.y;
  }
  return core;
}

vertex OpaqueQuadVertexOutput opaque_quad_vertex(
    uint unit_vertex_id [[vertex_id]], uint instance_id [[instance_id]],
    constant float2 *unit_vertices [[buffer(QuadInputIndex_Vertices)]],
    constant Quad *quads [[buffer(QuadInputIndex_Quads)]],
    constant uint *quad_indices [[buffer(QuadInputIndex_QuadIndices)]],
    constant Size_DevicePixels *viewport_size
    [[buffer(QuadInputIndex_ViewportSize)]]) {
  uint quad_id = quad_indices[instance_id];
  Quad quad = quads[quad_id];
  Bounds_ScaledPixels core = opaque_quad_core(quad);

  if (core.size.width <= 0.0 || core.size.height <= 0.0) {
    // Zero-area triangles are rejected before rasterization.
    return OpaqueQuadVertexOutput{float4(0.0, 0.0, 0.0, 1.0), float4(0.0)};
  }

  float2 unit_vertex = unit_vertices[unit_vertex_id];
  ClippedVertex clipped =
      clip_to_mask(unit_vertex, core, quad.content_mask.bounds);
  float4 device_position =
      to_device_position_impl(clipped.position, viewport_size);
  device_position.z = quad_depth(quad_id);
  // Matches the blended quad shader's solid-background output exactly: tag-0
  // quads resolve to `hsla_to_rgba(solid)`, and alpha-1 blending is an
  // overwrite.
  float4 color = hsla_to_rgba(quad.background.solid);

  return OpaqueQuadVertexOutput{device_position, color};
}

fragment float4 opaque_quad_fragment(OpaqueQuadFragmentInput input
                                     [[stage_in]]) {
  return input.color;
}

struct PreparedBlurredRoundedRect {
  float4 shape;
  float4 r1;
  float4 exponent;
  float4 scale;
};

struct ShadowVertexOutput {
  uint rounded_clip [[flat]];
  float4 position [[position]];
  float4 color [[flat]];
  uint shadow_id [[flat]];
  float4 blur_shape [[flat]];
  float4 blur_r1 [[flat]];
  float4 blur_exponent [[flat]];
  float4 blur_scale [[flat]];
};

struct ShadowFragmentInput {
  uint rounded_clip [[flat]];
  float4 position [[position]];
  float4 color [[flat]];
  uint shadow_id [[flat]];
  float4 blur_shape [[flat]];
  float4 blur_r1 [[flat]];
  float4 blur_exponent [[flat]];
  float4 blur_scale [[flat]];
};

float erf7(float x) {
  x = clamp(x, -10., 10.) * (2. / sqrt(M_PI_F));
  float xx = x * x;
  x = x + (0.24295 + (0.03395 + 0.0104 * xx) * xx) * (x * xx);
  return x / sqrt(1. + x * x);
}

// Raph Levien's blurred rounded rectangle approximation, whose blur s is sigma * sqrt(2):
// https://raphlinus.github.io/graphics/2020/04/21/blurred-rounded-rects.html
//
// Everything here depends only on the shadow, so it runs once per vertex and
// reaches the fragment shader flat.
PreparedBlurredRoundedRect prepare_blurred_rounded_rect(float2 size, Corners_ScaledPixels corner_radii, float sigma) {
  float s = max(sigma * sqrt(2.), 1e-6);
  float s_inv = 1. / s;
  float min_edge = min(size.x, size.y);
  float r_max = 0.5 * min_edge;
  float4 radii = float4(corner_radii.top_left, corner_radii.top_right,
                        corner_radii.bottom_right, corner_radii.bottom_left);
  float4 r0 = min(sqrt(radii * radii + 1.15 * 1.15 * s * s), r_max);
  float4 r1 = min(sqrt(radii * radii + 4. * s * s), r_max);

  float2 half_extent = 0.5 * s_inv * size;
  float delta = 1.25 * s * (exp(-half_extent.x * half_extent.x) - exp(-half_extent.y * half_extent.y));
  float2 adjusted_size = float2(size.x + min(delta, 0.), size.y - max(delta, 0.));
  float4 peak = s_inv * 0.5 * (max(adjusted_size.x, adjusted_size.y) - 0.5 * radii);

  PreparedBlurredRoundedRect output;
  output.shape = float4(0.5 * adjusted_size, s_inv, min_edge);
  output.r1 = r1;
  output.exponent = 2. * r1 / r0;
  output.scale = 0.5 * float4(erf7(peak.x), erf7(peak.y), erf7(peak.z), erf7(peak.w));
  return output;
}

float pick_corner(float2 center_to_point, float4 values) {
  Corners_ScaledPixels corners = {values.x, values.y, values.z, values.w};
  return pick_corner_radius(center_to_point, corners);
}

float blurred_rounded_rect(float2 center_to_point, PreparedBlurredRoundedRect prepared) {
  float r1 = pick_corner(center_to_point, prepared.r1);
  float exponent = pick_corner(center_to_point, prepared.exponent);
  float scale = pick_corner(center_to_point, prepared.scale);
  float s_inv = prepared.shape.z;
  float min_edge = prepared.shape.w;

  float2 p0 = abs(center_to_point) - (prepared.shape.xy - r1);
  float2 p1 = max(p0, 0.);
  float d_pos;
  if (p1.x > 0. && p1.y > 0.) {
    d_pos = pow(pow(p1.x, exponent) + pow(p1.y, exponent), 1. / exponent);
  } else {
    d_pos = p1.x + p1.y;
  }

  float d_neg = min(max(p0.x, p0.y), 0.);
  float d = d_pos + d_neg - r1;
  return scale * (erf7(s_inv * (min_edge + d)) - erf7(s_inv * d));
}

vertex ShadowVertexOutput shadow_vertex(
    uint unit_vertex_id [[vertex_id]], uint shadow_id [[instance_id]],
    constant float2 *unit_vertices [[buffer(ShadowInputIndex_Vertices)]],
    constant Shadow *shadows [[buffer(ShadowInputIndex_Shadows)]],
    constant Size_DevicePixels *viewport_size
    [[buffer(ShadowInputIndex_ViewportSize)]]) {
  float2 unit_vertex = unit_vertices[unit_vertex_id];
  Shadow shadow = shadows[shadow_id];

  Bounds_ScaledPixels bounds;
  if (shadow.inset != 0u) {
    bounds = shadow.element_bounds;
  } else {
    // Leave room for the tail of the blur outside the shadow rect.
    float margin = 3. * shadow.blur_radius;
    bounds = shadow.bounds;
    bounds.origin.x -= margin;
    bounds.origin.y -= margin;
    bounds.size.width += 2. * margin;
    bounds.size.height += 2. * margin;
  }

  ClippedVertex clipped =
      clip_to_mask(unit_vertex, bounds, shadow.content_mask.bounds);
  float4 device_position =
      to_device_position_impl(clipped.position, viewport_size);
  float4 color = hsla_to_rgba(shadow.color);
  PreparedBlurredRoundedRect blur = prepare_blurred_rounded_rect(
      float2(shadow.bounds.size.width, shadow.bounds.size.height),
      shadow.corner_radii, shadow.blur_radius);

  return ShadowVertexOutput{
      shadow.rounded_clip,
      device_position,
      color,
      shadow_id,
      blur.shape,
      blur.r1,
      blur.exponent,
      blur.scale};
}

fragment float4 shadow_fragment(ShadowFragmentInput input [[stage_in]],
                                constant Shadow *shadows
                                [[buffer(ShadowInputIndex_Shadows)]],
    constant RoundedClipNode *rounded_masks [[buffer(15)]]) {
  Shadow shadow = shadows[input.shadow_id];

  float alpha;
  if (shadow.blur_radius == 0.) {
    float distance = quad_sdf(input.position.xy, shadow.bounds, shadow.corner_radii);
    alpha = saturate(0.5 - distance);
  } else {
    float2 origin = float2(shadow.bounds.origin.x, shadow.bounds.origin.y);
    float2 size = float2(shadow.bounds.size.width, shadow.bounds.size.height);
    float2 center = origin + size / 2.;
    PreparedBlurredRoundedRect blur = {input.blur_shape, input.blur_r1,
                                       input.blur_exponent, input.blur_scale};
    alpha = blurred_rounded_rect(input.position.xy - center, blur);
  }

  if (shadow.inset != 0u) {
    // The inset shadow is the complement of the (blurred) hole rect, clipped to the element.
    // `saturate(0.5 - d)` gives a 1-pixel antialiased edge: d <= -0.5 -> 1, d >= 0.5 -> 0.
    alpha = 1. - alpha;
    float element_distance = quad_sdf(input.position.xy, shadow.element_bounds,
                                      shadow.element_corner_radii);
    alpha *= saturate(0.5 - element_distance);
  }

  return (input.color * float4(1., 1., 1., alpha)) * float4(1.0, 1.0, 1.0, rounded_clip_coverage(input.rounded_clip, input.position.xy, rounded_masks));
}

struct BackdropBlurVertexOutput {
  float4 position [[position]];
  uint blur_id [[flat]];
};

struct BackdropBlurFragmentInput {
  float4 position [[position]];
  uint blur_id [[flat]];
};

vertex BackdropBlurVertexOutput backdrop_blur_vertex(
    uint unit_vertex_id [[vertex_id]], uint blur_id [[instance_id]],
    constant float2 *unit_vertices [[buffer(BackdropBlurInputIndex_Vertices)]],
    constant BackdropBlur *blurs [[buffer(BackdropBlurInputIndex_Blurs)]],
    constant Size_DevicePixels *viewport_size
    [[buffer(BackdropBlurInputIndex_ViewportSize)]]) {
  float2 unit_vertex = unit_vertices[unit_vertex_id];
  BackdropBlur blur = blurs[blur_id];
  ClippedVertex clipped =
      clip_to_mask(unit_vertex, blur.bounds, blur.content_mask.bounds);
  float4 device_position =
      to_device_position_impl(clipped.position, viewport_size);
  return BackdropBlurVertexOutput{
      device_position,
      blur_id};
}

struct BackdropBlurPassVertexOutput {
  float4 position [[position]];
};

// The viewport of a blur pass is exactly the scratch rect it writes, so the
// unit square spans it.
vertex BackdropBlurPassVertexOutput backdrop_blur_pass_vertex(
    uint unit_vertex_id [[vertex_id]],
    constant float2 *unit_vertices [[buffer(BackdropBlurPassInputIndex_Vertices)]]) {
  float2 unit_vertex = unit_vertices[unit_vertex_id];
  return BackdropBlurPassVertexOutput{
      float4(unit_vertex.x * 2. - 1., 1. - unit_vertex.y * 2., 0., 1.)};
}

// Where along one axis the first pass reads for scratch texel `texel`, as
// (first sample, step between samples, sample count). Texel 0 and the last
// texel are a pad holding the region's own outermost pixel row or column,
// so the gaussian, which clamps at the scratch's edges, repeats the region's
// edge pixels as an exact clamp at full resolution would, not the average of
// the outermost block. Every texel between stands for a block of
// `downsample` pixels, read at the corners shared by pairs of pixels so each
// linear sample averages two of them.
float3 backdrop_blur_block_samples(uint texel, uint last, float origin, float low, float high,
                                   float downsample) {
  if (texel == 0u) {
    return float3(low, 0., 1.);
  }
  if (texel == last) {
    return float3(high, 0., 1.);
  }
  float block = origin + float(texel - 1u) * downsample;
  if (downsample < 2.) {
    return float3(block + 0.5, 0., 1.);
  }
  return float3(block + 1., 2., downsample * 0.5);
}

// The first pass: the region of the target being drawn, box-averaged down
// into the scratch, `downsample` source pixels per texel on each axis, inside
// a one-texel pad of edge pixels (see `backdrop_blur_block_samples`). Samples
// are clamped to the region, so a block that runs past its edge repeats the
// edge pixels, and nothing outside the surface bleeds in across its outline.
fragment float4 backdrop_blur_downsample_fragment(
    BackdropBlurPassVertexOutput input [[stage_in]],
    constant BackdropBlurPass *pass [[buffer(BackdropBlurPassInputIndex_Pass)]],
    texture2d<float> source [[texture(BackdropBlurPassInputIndex_Source)]]) {
  constexpr sampler pixels(coord::pixel, filter::linear, address::clamp_to_edge);
  float downsample = pass->downsample;
  float2 origin = float2(pass->source_origin.x, pass->source_origin.y);
  float2 low = origin + 0.5;
  float2 high = origin + float2(pass->source_size.x, pass->source_size.y) - 0.5;
  uint2 texel = uint2(input.position.xy);
  uint2 last = uint2(pass->scratch_size.x, pass->scratch_size.y) - 1u;
  float3 columns = backdrop_blur_block_samples(texel.x, last.x, origin.x, low.x, high.x, downsample);
  float3 rows = backdrop_blur_block_samples(texel.y, last.y, origin.y, low.y, high.y, downsample);
  float4 sum = float4(0.);
  for (uint row = 0; row < uint(rows.z); row++) {
    float y = clamp(rows.x + float(row) * rows.y, low.y, high.y);
    for (uint column = 0; column < uint(columns.z); column++) {
      float x = clamp(columns.x + float(column) * columns.y, low.x, high.x);
      sum += source.sample(pixels, float2(x, y));
    }
  }
  return sum / (rows.z * columns.z);
}

// One half of the separable gaussian, between two scratch textures, along
// `axis`. `taps` pairs neighbouring texels into one linear sample each
// (offset, weight); the first is the centre. Clamping each sample to the
// outermost texel centres repeats the edge, as the region's own edges are.
float4 backdrop_blur_along(float2 axis, float2 position, constant BackdropBlurPass *pass,
                           constant float2 *taps, texture2d<float> source) {
  constexpr sampler pixels(coord::pixel, filter::linear, address::clamp_to_edge);
  float2 high = float2(pass->scratch_size.x, pass->scratch_size.y) - 0.5;
  float4 sum = taps[0].y * source.sample(pixels, position);
  for (uint tap = 1; tap <= pass->taps; tap++) {
    float2 offset = axis * taps[tap].x;
    sum += taps[tap].y * (source.sample(pixels, clamp(position - offset, float2(0.5), high)) +
                          source.sample(pixels, clamp(position + offset, float2(0.5), high)));
  }
  return sum;
}

fragment float4 backdrop_blur_horizontal_fragment(
    BackdropBlurPassVertexOutput input [[stage_in]],
    constant BackdropBlurPass *pass [[buffer(BackdropBlurPassInputIndex_Pass)]],
    constant float2 *taps [[buffer(BackdropBlurPassInputIndex_Taps)]],
    texture2d<float> source [[texture(BackdropBlurPassInputIndex_Source)]]) {
  return backdrop_blur_along(float2(1., 0.), input.position.xy, pass, taps, source);
}

fragment float4 backdrop_blur_vertical_fragment(
    BackdropBlurPassVertexOutput input [[stage_in]],
    constant BackdropBlurPass *pass [[buffer(BackdropBlurPassInputIndex_Pass)]],
    constant float2 *taps [[buffer(BackdropBlurPassInputIndex_Taps)]],
    texture2d<float> source [[texture(BackdropBlurPassInputIndex_Source)]]) {
  return backdrop_blur_along(float2(0., 1.), input.position.xy, pass, taps, source);
}

struct BackdropBlurFragmentOutput {
  float4 color [[color(0), index(0)]];
  float4 keep [[color(0), index(1)]];
};

// Dual-source blending turns `color + destination * (1 - keep.a)` into
// `mix(destination, frosted, coverage)`: wherever the effect is partial (the
// anti-aliased rounded edge, a fade) the framebuffer pixel itself shows
// through, alpha included, without a copy of the region to read it from.
// The blurred scratch is upsampled with linear filtering. Scratch texel
// i + 1's centre stands for the centre of block i, at source position
// `source_origin + downsample * (i + 0.5)`, and the pads either side stand
// for the half block beyond each edge, so a pixel between the outermost
// block's centre and the region's edge falls between that block and the pad.
fragment BackdropBlurFragmentOutput backdrop_blur_fragment(
    BackdropBlurFragmentInput input [[stage_in]],
    constant BackdropBlur *blurs [[buffer(BackdropBlurInputIndex_Blurs)]],
    constant BackdropBlurPass *pass [[buffer(BackdropBlurInputIndex_Pass)]],
    texture2d<float> blurred [[texture(BackdropBlurInputIndex_Blurred)]]) {
  constexpr sampler pixels(coord::pixel, filter::linear, address::clamp_to_edge);
  BackdropBlur blur = blurs[input.blur_id];
  float2 origin = float2(pass->source_origin.x, pass->source_origin.y);
  float2 high = float2(pass->scratch_size.x, pass->scratch_size.y) - 0.5;
  float2 position = clamp((input.position.xy - origin) / pass->downsample + 1., float2(0.5), high);
  float4 frosted = blurred.sample(pixels, position);

  float luma = dot(frosted.rgb, float3(0.2126, 0.7152, 0.0722));
  frosted.rgb = clamp(mix(float3(luma), frosted.rgb, blur.saturation), 0., frosted.a);

  float distance = quad_sdf(input.position.xy, blur.bounds, blur.corner_radii);
  float coverage = saturate(0.5 - distance) * saturate(blur.opacity);
  return BackdropBlurFragmentOutput{frosted * coverage, float4(coverage)};
}

struct UnderlineVertexOutput {
  uint rounded_clip [[flat]];
  float4 position [[position]];
  float4 color [[flat]];
  uint underline_id [[flat]];
};

struct UnderlineFragmentInput {
  uint rounded_clip [[flat]];
  float4 position [[position]];
  float4 color [[flat]];
  uint underline_id [[flat]];
};

vertex UnderlineVertexOutput underline_vertex(
    uint unit_vertex_id [[vertex_id]], uint underline_id [[instance_id]],
    constant float2 *unit_vertices [[buffer(UnderlineInputIndex_Vertices)]],
    constant Underline *underlines [[buffer(UnderlineInputIndex_Underlines)]],
    constant Size_DevicePixels *viewport_size
    [[buffer(ShadowInputIndex_ViewportSize)]]) {
  float2 unit_vertex = unit_vertices[unit_vertex_id];
  Underline underline = underlines[underline_id];
  ClippedVertex clipped = clip_to_mask(unit_vertex, underline.bounds,
                                       underline.content_mask.bounds);
  float4 device_position =
      to_device_position_impl(clipped.position, viewport_size);
  float4 color = hsla_to_rgba(underline.color);
  return UnderlineVertexOutput{
      underline.rounded_clip,
      device_position,
      color,
      underline_id | (underline.wavy == 0u ? SIMPLE_PRIMITIVE_FLAG : 0u)};
}

fragment float4 underline_fragment(UnderlineFragmentInput input [[stage_in]],
                                   constant Underline *underlines
                                   [[buffer(UnderlineInputIndex_Underlines)]],
    constant RoundedClipNode *rounded_masks [[buffer(15)]]) {
  const float WAVE_FREQUENCY = 2.0;
  const float WAVE_HEIGHT_RATIO = 0.8;

  if ((input.underline_id & SIMPLE_PRIMITIVE_FLAG) != 0u) {
    return (input.color) * float4(1.0, 1.0, 1.0, rounded_clip_coverage(input.rounded_clip, input.position.xy, rounded_masks));
  }

  Underline underline = underlines[input.underline_id];
  float half_thickness = underline.thickness * 0.5;
  float2 origin =
      float2(underline.bounds.origin.x, underline.bounds.origin.y);

  float2 st = ((input.position.xy - origin) / underline.bounds.size.height) -
              float2(0., 0.5);
  float frequency = (M_PI_F * WAVE_FREQUENCY * underline.thickness) / underline.bounds.size.height;
  float amplitude = (underline.thickness * WAVE_HEIGHT_RATIO) / underline.bounds.size.height;

  float sine = sin(st.x * frequency) * amplitude;
  float dSine = cos(st.x * frequency) * amplitude * frequency;
  float distance = (st.y - sine) / sqrt(1. + dSine * dSine);
  float distance_in_pixels = distance * underline.bounds.size.height;
  float distance_from_top_border = distance_in_pixels - half_thickness;
  float distance_from_bottom_border = distance_in_pixels + half_thickness;
  float alpha = saturate(
      0.5 - max(-distance_from_bottom_border, distance_from_top_border));
  return (input.color * float4(1., 1., 1., alpha)) * float4(1.0, 1.0, 1.0, rounded_clip_coverage(input.rounded_clip, input.position.xy, rounded_masks));
}

struct MonochromeSpriteVertexOutput {
  uint rounded_clip [[flat]];
  float4 position [[position]];
  float2 tile_position;
  float2 local_position;
  float4 color [[flat]];
  uint sprite_id [[flat]];
  uint effect_kind [[flat]];
  float4 clip_distance;
};

struct MonochromeSpriteFragmentInput {
  uint rounded_clip [[flat]];
  float4 position [[position]];
  float2 tile_position;
  float2 local_position;
  float4 color [[flat]];
  uint sprite_id [[flat]];
  uint effect_kind [[flat]];
  float4 clip_distance;
};

// Highlight profile of the shimmer band. `u` is the position across the band
// along `effect.direction` — projecting onto that vector is what makes the band
// diagonal rather than axis-aligned. Each side of `effect.peak` gets its own
// smoothstep ramp, so moving the peak off center gives the band a sharp leading
// edge and a long trailing tail; `effect.falloff` shapes those ramps and
// `effect.core_gain` holds light back from the shoulders so the core reads
// brighter than the rest of the band.
//
// Callers must check `kind` first (via the flat `effect_kind` varying) so
// unshimmered text never reaches this function or the storage read it needs.
float sprite_effect_intensity(SpriteEffect effect, float2 local_position) {
  float2 direction = float2(effect.direction[0], effect.direction[1]);
  float2 offset = local_position - float2(effect.origin.x, effect.origin.y);
  float width = max(effect.band_width, 1.0);
  float u = (dot(offset, direction) - effect.band_origin) / width;
  if (u <= 0.0 || u >= 1.0) {
    return 0.0;
  }
  float side = u < effect.peak ? u / effect.peak : (1.0 - u) / (1.0 - effect.peak);
  float ramp = side * side * (3.0 - 2.0 * side);
  ramp = pow(ramp, effect.falloff);
  float core = saturate(1.0 - fabs(u - effect.peak) / effect.core_spread);
  core = core * core * (3.0 - 2.0 * core);
  return ramp * (1.0 - effect.core_gain * (1.0 - core));
}

vertex MonochromeSpriteVertexOutput monochrome_sprite_vertex(
    uint unit_vertex_id [[vertex_id]], uint sprite_id [[instance_id]],
    constant float2 *unit_vertices [[buffer(SpriteInputIndex_Vertices)]],
    constant MonochromeSprite *sprites [[buffer(SpriteInputIndex_Sprites)]],
    constant Size_DevicePixels *viewport_size
    [[buffer(SpriteInputIndex_ViewportSize)]],
    constant Size_DevicePixels *atlas_size
    [[buffer(SpriteInputIndex_AtlasTextureSize)]]) {
  float2 unit_vertex = unit_vertices[unit_vertex_id];
  MonochromeSprite sprite = sprites[sprite_id];
  float4 device_position =
      to_device_position_transformed(unit_vertex, sprite.bounds, sprite.transformation, viewport_size);
  float4 clip_distance = distance_from_clip_rect_transformed(unit_vertex, sprite.bounds,
                                                 sprite.content_mask.bounds, sprite.transformation);
  float2 tile_position = to_tile_position(unit_vertex, sprite.tile, atlas_size);
  float2 local_position = float2(
      sprite.bounds.origin.x + unit_vertex.x * sprite.bounds.size.width,
      sprite.bounds.origin.y + unit_vertex.y * sprite.bounds.size.height);
  float4 color = hsla_to_rgba(sprite.color);
  return MonochromeSpriteVertexOutput{
      sprite.rounded_clip,
      device_position,
      tile_position,
      local_position,
      color,
      sprite_id,
      sprite.effect.kind,
      {clip_distance.x, clip_distance.y, clip_distance.z, clip_distance.w}};
}

fragment float4 monochrome_sprite_fragment(
    MonochromeSpriteFragmentInput input [[stage_in]],
    constant MonochromeSprite *sprites [[buffer(SpriteInputIndex_Sprites)]],
    texture2d<float> atlas_texture [[texture(SpriteInputIndex_AtlasTexture)]],
    constant RoundedClipNode *rounded_masks [[buffer(15)]]) {
  if (any(input.clip_distance < float4(0.0))) {
    return (float4(0.0)) * float4(1.0, 1.0, 1.0, rounded_clip_coverage(input.rounded_clip, input.position.xy, rounded_masks));
  }

  constexpr sampler atlas_texture_sampler(mag_filter::linear,
                                          min_filter::linear);
  float4 sample =
      atlas_texture.sample(atlas_texture_sampler, input.tile_position);
  float4 base_color = input.color;

  // Virtually every glyph in the window is unshimmered, so bail out before
  // touching the sprite storage buffer at all. Reading `sprites[sprite_id]`
  // from the fragment stage is a dependent memory load per fragment, and the
  // overlay blend below ends in a divide; neither is worth paying on all text
  // for an effect a handful of labels use.
  if (input.effect_kind != SPRITE_EFFECT_SHIMMER) {
    return (float4(base_color.rgb, base_color.a * sample.a)) * float4(1.0, 1.0, 1.0, rounded_clip_coverage(input.rounded_clip, input.position.xy, rounded_masks));
  }

  float intensity = sprite_effect_intensity(
      sprites[input.sprite_id].effect,
      input.local_position);
  float4 highlight_color = hsla_to_rgba(sprites[input.sprite_id].effect.highlight_color);
  float glyph_alpha = sample.a;
  float base_alpha = base_color.a * glyph_alpha;
  float overlay_alpha = highlight_color.a * intensity * glyph_alpha;
  float combined_alpha = overlay_alpha + base_alpha * (1.0 - overlay_alpha);
  float3 combined_rgb = base_color.rgb;
  if (combined_alpha > 0.0) {
    combined_rgb =
        (highlight_color.rgb * overlay_alpha +
         base_color.rgb * base_alpha * (1.0 - overlay_alpha)) /
        combined_alpha;
  }
  return (float4(combined_rgb, combined_alpha)) * float4(1.0, 1.0, 1.0, rounded_clip_coverage(input.rounded_clip, input.position.xy, rounded_masks));
}

struct PolychromeSpriteVertexOutput {
  uint rounded_clip [[flat]];
  float4 position [[position]];
  float2 tile_position;
  uint sprite_id [[flat]];
  float opacity [[flat]];
};

struct PolychromeSpriteFragmentInput {
  uint rounded_clip [[flat]];
  float4 position [[position]];
  float2 tile_position;
  uint sprite_id [[flat]];
  float opacity [[flat]];
};

vertex PolychromeSpriteVertexOutput polychrome_sprite_vertex(
    uint unit_vertex_id [[vertex_id]], uint sprite_id [[instance_id]],
    constant float2 *unit_vertices [[buffer(SpriteInputIndex_Vertices)]],
    constant PolychromeSprite *sprites [[buffer(SpriteInputIndex_Sprites)]],
    constant Size_DevicePixels *viewport_size
    [[buffer(SpriteInputIndex_ViewportSize)]],
    constant Size_DevicePixels *atlas_size
    [[buffer(SpriteInputIndex_AtlasTextureSize)]]) {

  float2 unit_vertex = unit_vertices[unit_vertex_id];
  PolychromeSprite sprite = sprites[sprite_id];
  ClippedVertex clipped =
      clip_to_mask(unit_vertex, sprite.bounds, sprite.content_mask.bounds);
  float4 device_position =
      to_device_position_impl(clipped.position, viewport_size);
  float2 tile_position =
      to_tile_position(clipped.unit_vertex, sprite.tile, atlas_size);
  bool simple = sprite.grayscale == 0u &&
                sprite.corner_radii.top_left == 0.0 &&
                sprite.corner_radii.top_right == 0.0 &&
                sprite.corner_radii.bottom_right == 0.0 &&
                sprite.corner_radii.bottom_left == 0.0;
  return PolychromeSpriteVertexOutput{
      sprite.rounded_clip,
      device_position,
      tile_position,
      sprite_id | (simple ? SIMPLE_PRIMITIVE_FLAG : 0u),
      sprite.opacity};
}

fragment float4 polychrome_sprite_fragment(
    PolychromeSpriteFragmentInput input [[stage_in]],
    constant PolychromeSprite *sprites [[buffer(SpriteInputIndex_Sprites)]],
    texture2d<float> atlas_texture [[texture(SpriteInputIndex_AtlasTexture)]],
    constant RoundedClipNode *rounded_masks [[buffer(15)]]) {
  constexpr sampler atlas_texture_sampler(mag_filter::linear,
                                          min_filter::linear);
  float4 color =
      atlas_texture.sample(atlas_texture_sampler, input.tile_position);
  if ((input.sprite_id & SIMPLE_PRIMITIVE_FLAG) != 0u) {
    return float4(color.rgb, color.a * input.opacity) * float4(1.0, 1.0, 1.0, rounded_clip_coverage(input.rounded_clip, input.position.xy, rounded_masks));
  }

  PolychromeSprite sprite = sprites[input.sprite_id];
  if (sprite.grayscale) {
    float grayscale = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
    color.r = grayscale;
    color.g = grayscale;
    color.b = grayscale;
  }
  float distance =
      quad_sdf(input.position.xy, sprite.bounds, sprite.corner_radii);
  color.a *= input.opacity * saturate(0.5 - distance);
  return (color) * float4(1.0, 1.0, 1.0, rounded_clip_coverage(input.rounded_clip, input.position.xy, rounded_masks));
}

struct PathTileFragmentInput {
  uint rounded_clip [[flat]];
  float4 position [[position]];
  uint tile_id [[flat]];
  float4 curve_box [[flat]];
  uint background_tag [[flat]];
  float4 background_solid [[flat]];
  float4 background_color0 [[flat]];
  float4 background_color1 [[flat]];
};

vertex PathTileFragmentInput path_tile_vertex(
    uint unit_vertex_id [[vertex_id]], uint tile_id [[instance_id]],
    constant float2 *unit_vertices [[buffer(PathTileInputIndex_Vertices)]],
    constant PathTile *tiles [[buffer(PathTileInputIndex_Tiles)]],
    constant PathPaint *paints [[buffer(PathTileInputIndex_Paints)]],
    constant Size_DevicePixels *viewport_size
    [[buffer(PathTileInputIndex_ViewportSize)]]) {
  float2 unit_vertex = unit_vertices[unit_vertex_id];
  PathTile tile = tiles[tile_id];
  PathPaint paint = paints[tile.paint];

  float2 mask_top_left = float2(paint.content_mask.bounds.origin.x,
                                paint.content_mask.bounds.origin.y);
  float2 mask_bottom_right =
      mask_top_left + float2(paint.content_mask.bounds.size.width,
                             paint.content_mask.bounds.size.height);
  float2 top_left = clamp(float2(tile.top_left.x, tile.top_left.y),
                          mask_top_left, mask_bottom_right);
  float2 bottom_right = clamp(float2(tile.bottom_right.x, tile.bottom_right.y),
                              top_left, mask_bottom_right);
  // Neighbouring tiles store the edge they share as the same number, so
  // each corner is selected rather than interpolated: `a + (b - a)` is not
  // always `b`, and edges that differ in their last bit leave a seam.
  float2 position = float2(unit_vertex.x != 0.0 ? bottom_right.x : top_left.x,
                           unit_vertex.y != 0.0 ? bottom_right.y : top_left.y);

  GradientColor gradient = prepare_fill_color(
    paint.color.tag,
    paint.color.color_space,
    paint.color.solid,
    paint.color.colors[0].color,
    paint.color.colors[1].color
  );

  return PathTileFragmentInput{
      paint.rounded_clip,
      to_device_position_impl(position, viewport_size),
      tile_id,
      float4(tile.curves_top_left.x, tile.curves_top_left.y,
             tile.curves_bottom_right.x, tile.curves_bottom_right.y),
      paint.color.tag,
      gradient.solid,
      gradient.color0,
      gradient.color1};
}

// Solves a*t^2 + b*t + c = 0 for the root in [0, 1] on a monotone piece.
float monotone_quadratic_root(float a, float b, float c, float direction) {
  float discriminant = max(b * b - 4.0 * a * c, 0.0);
  float denominator = b + direction * sqrt(discriminant);
  if (denominator == 0.0) {
    return 0.0;
  }
  return saturate(-2.0 * c / denominator);
}

// Coverage estimate adapted from https://github.com/EricLengyel/Slug/blob/be3c13eb7d63f9e8aa5c583e42d92c374cb91d98/SlugPixelShader.hlsl
// Copyright (c) 2017 Eric Lengyel.
// Licensed under the MIT license.

// Accumulates the curve's winding along the pixel's leftward and upward rays.
// Like Slug, a crossing within half a pixel of the center counts fractionally.
void add_curve_winding(PathCurve curve, float2 center, thread float2 &winding,
                       thread float2 &weight) {
  float2 p0 = float2(curve.p0.x, curve.p0.y);
  float2 p1 = float2(curve.p1.x, curve.p1.y);
  float2 a = float2(curve.a.x, curve.a.y);
  float2 b = float2(curve.b.x, curve.b.y);
  float min_x = min(p0.x, p1.x);
  float max_x = max(p0.x, p1.x);

  if (p0.y <= center.y && center.y < p1.y) {
    if (max_x <= center.x - 0.5) {
      winding.x += curve.sign;
    } else {
      float t = monotone_quadratic_root(a.y, b.y, p0.y - center.y, 1.0);
      float x = (a.x * t + b.x) * t + p0.x;
      float distance = center.x - x;
      winding.x += curve.sign * saturate(distance + 0.5);
      weight.x = max(weight.x, saturate(1.0 - 2.0 * abs(distance)));
    }
  }

  if (min_x <= center.x && center.x < max_x) {
    float direction = p1.x > p0.x ? 1.0 : -1.0;
    if (p1.y <= center.y - 0.5) {
      winding.y -= curve.sign * direction;
    } else if (p0.y < center.y + 0.5) {
      float t = monotone_quadratic_root(a.x, b.x, p0.x - center.x, direction);
      float y = (a.y * t + b.y) * t + p0.y;
      float distance = center.y - y;
      winding.y -= curve.sign * direction * saturate(distance + 0.5);
      weight.y = max(weight.y, saturate(1.0 - 2.0 * abs(distance)));
    }
  }
}

// The backdrop gives the unlisted curves' winding along a leftward ray from
// the tile's reference corner. We need it at the current pixel, along both
// leftward and upward rays. Because paths are closed, chains of unlisted
// curves can only begin or end where a listed curve does. So, we can compute
// the winding at the current pixel by correcting the backdrop for each listed
// endpoint between its ray and the pixel.
void correct_for_endpoint(float2 endpoint, float step, float2 reference,
                          float2 center, thread float2 &winding) {
  if (endpoint.x <= reference.x) {
    winding.x -= step * (float(endpoint.y <= center.y) -
                         float(endpoint.y <= reference.y));
  }
  if (endpoint.y <= reference.y && endpoint.x <= center.x) {
    winding.y += step;
  }
}

fragment float4 path_tile_fragment(
    PathTileFragmentInput input [[stage_in]],
    constant PathTile *tiles [[buffer(PathTileInputIndex_Tiles)]],
    constant PathCurve *curves [[buffer(PathTileInputIndex_Curves)]],
    constant PathPaint *paints [[buffer(PathTileInputIndex_Paints)]],
    constant RoundedClipNode *rounded_masks [[buffer(15)]]) {
  float2 center = floor(input.position.xy) + 0.5;
  // A solid fill well inside the path needs neither its curves nor its paint.
  if (input.background_tag == 0u &&
      (any(center < input.curve_box.xy) || any(center > input.curve_box.zw))) {
    return input.background_solid * float4(1.0, 1.0, 1.0, rounded_clip_coverage(input.rounded_clip, input.position.xy, rounded_masks));
  }
  PathTile tile = tiles[input.tile_id];
  PathPaint paint = paints[tile.paint];
  float2 reference = float2(tile.reference.x, tile.reference.y);

  float2 winding = float2(float(tile.backdrop));
  float2 weight = float2(0.0);
  for (uint i = 0u; i < tile.curve_count; i++) {
    PathCurve curve = curves[tile.curve_start + i];
    float2 p0 = float2(curve.p0.x, curve.p0.y);
    float2 p1 = float2(curve.p1.x, curve.p1.y);
    if (min(p0.x, p1.x) >= center.x + 0.5) {
      break;
    }
    if (p0.y < center.y + 0.5) {
      add_curve_winding(curve, center, winding, weight);
      correct_for_endpoint(p0, curve.sign, reference, center, winding);
      correct_for_endpoint(p1, -curve.sign, reference, center, winding);
    }
  }

  // Weight the winding computed by each ray by its nearest crossing's distance
  // from the pixel center, using Slug's technique.
  float coverage =
      max(abs(dot(winding, weight)) / max(weight.x + weight.y, 1.0 / 65536.0),
          min(abs(winding.x), abs(winding.y)));
  coverage = paint.even_odd != 0u
                 ? 1.0 - abs(1.0 - fract(coverage * 0.5) * 2.0)
                 : saturate(coverage);

  if (coverage == 0.0) {
    return float4(0.0);
  }

  float4 color = fill_color(paint.color, input.position.xy, paint.bounds,
                            input.background_solid, input.background_color0,
                            input.background_color1);
  return float4(color.rgb, color.a * coverage) * float4(1.0, 1.0, 1.0, rounded_clip_coverage(input.rounded_clip, input.position.xy, rounded_masks));
}

struct SurfaceVertexOutput {
  uint rounded_clip [[flat]];
  float4 position [[position]];
  float2 texture_position;
};

struct SurfaceFragmentInput {
  uint rounded_clip [[flat]];
  float4 position [[position]];
  float2 texture_position;
};

vertex SurfaceVertexOutput surface_vertex(
    uint unit_vertex_id [[vertex_id]], uint surface_id [[instance_id]],
    constant float2 *unit_vertices [[buffer(SurfaceInputIndex_Vertices)]],
    constant SurfaceBounds *surfaces [[buffer(SurfaceInputIndex_Surfaces)]],
    constant Size_DevicePixels *viewport_size
    [[buffer(SurfaceInputIndex_ViewportSize)]],
    constant Size_DevicePixels *texture_size
    [[buffer(SurfaceInputIndex_TextureSize)]]) {
  float2 unit_vertex = unit_vertices[unit_vertex_id];
  SurfaceBounds surface = surfaces[surface_id];
  ClippedVertex clipped =
      clip_to_mask(unit_vertex, surface.bounds, surface.content_mask.bounds);
  float4 device_position =
      to_device_position_impl(clipped.position, viewport_size);
  // We are going to copy the whole texture, so the texture position corresponds
  // to the vertex's position within the unclipped surface bounds.
  float2 texture_position = clipped.unit_vertex;
  return SurfaceVertexOutput{
      surface.rounded_clip,
      device_position,
      texture_position};
}

fragment float4 surface_fragment(SurfaceFragmentInput input [[stage_in]],
                                 texture2d<float> y_texture
                                 [[texture(SurfaceInputIndex_YTexture)]],
                                 texture2d<float> cb_cr_texture
                                 [[texture(SurfaceInputIndex_CbCrTexture)]],
    constant RoundedClipNode *rounded_masks [[buffer(15)]]) {
  constexpr sampler texture_sampler(mag_filter::linear, min_filter::linear);
  const float4x4 ycbcrToRGBTransform =
      float4x4(float4(+1.0000f, +1.0000f, +1.0000f, +0.0000f),
               float4(+0.0000f, -0.3441f, +1.7720f, +0.0000f),
               float4(+1.4020f, -0.7141f, +0.0000f, +0.0000f),
               float4(-0.7010f, +0.5291f, -0.8860f, +1.0000f));
  float4 ycbcr = float4(
      y_texture.sample(texture_sampler, input.texture_position).r,
      cb_cr_texture.sample(texture_sampler, input.texture_position).rg, 1.0);

  return (ycbcrToRGBTransform * ycbcr) * float4(1.0, 1.0, 1.0, rounded_clip_coverage(input.rounded_clip, input.position.xy, rounded_masks));
}

float4 hsla_to_rgba(Hsla hsla) {
  float h = hsla.h * 6.0; // Now, it's an angle but scaled in [0, 6) range
  float s = hsla.s;
  float l = hsla.l;
  float a = hsla.a;

  float c = (1.0 - fabs(2.0 * l - 1.0)) * s;
  float x = c * (1.0 - fabs(fmod(h, 2.0) - 1.0));
  float m = l - c / 2.0;

  float r = 0.0;
  float g = 0.0;
  float b = 0.0;

  if (h >= 0.0 && h < 1.0) {
    r = c;
    g = x;
    b = 0.0;
  } else if (h >= 1.0 && h < 2.0) {
    r = x;
    g = c;
    b = 0.0;
  } else if (h >= 2.0 && h < 3.0) {
    r = 0.0;
    g = c;
    b = x;
  } else if (h >= 3.0 && h < 4.0) {
    r = 0.0;
    g = x;
    b = c;
  } else if (h >= 4.0 && h < 5.0) {
    r = x;
    g = 0.0;
    b = c;
  } else {
    r = c;
    g = 0.0;
    b = x;
  }

  float4 rgba;
  rgba.x = (r + m);
  rgba.y = (g + m);
  rgba.z = (b + m);
  rgba.w = a;
  return rgba;
}

float3 srgb_to_linear(float3 color) {
  return pow(color, float3(2.2));
}

float3 linear_to_srgb(float3 color) {
  return pow(color, float3(1.0 / 2.2));
}

// Converts a sRGB color to the Oklab color space.
// Reference: https://bottosson.github.io/posts/oklab/#converting-from-linear-srgb-to-oklab
float4 srgb_to_oklab(float4 color) {
  // Convert non-linear sRGB to linear sRGB
  color = float4(srgb_to_linear(color.rgb), color.a);

  float l = 0.4122214708 * color.r + 0.5363325363 * color.g + 0.0514459929 * color.b;
  float m = 0.2119034982 * color.r + 0.6806995451 * color.g + 0.1073969566 * color.b;
  float s = 0.0883024619 * color.r + 0.2817188376 * color.g + 0.6299787005 * color.b;

  float l_ = pow(l, 1.0/3.0);
  float m_ = pow(m, 1.0/3.0);
  float s_ = pow(s, 1.0/3.0);

  return float4(
   	0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
   	1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
   	0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
   	color.a
  );
}

// Converts an Oklab color to the sRGB color space.
float4 oklab_to_srgb(float4 color) {
  float l_ = color.r + 0.3963377774 * color.g + 0.2158037573 * color.b;
  float m_ = color.r - 0.1055613458 * color.g - 0.0638541728 * color.b;
  float s_ = color.r - 0.0894841775 * color.g - 1.2914855480 * color.b;

  float l = l_ * l_ * l_;
  float m = m_ * m_ * m_;
  float s = s_ * s_ * s_;

  float3 linear_rgb = float3(
   	4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
   	-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
   	-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
  );

  // Convert linear sRGB to non-linear sRGB
  return float4(linear_to_srgb(linear_rgb), color.a);
}

float4 to_device_position_impl(float2 position,
                               constant Size_DevicePixels *input_viewport_size) {
  float2 viewport_size = float2((float)input_viewport_size->width,
                                (float)input_viewport_size->height);
  float2 device_position =
      position / viewport_size * float2(2., -2.) + float2(-1., 1.);
  return float4(device_position, 0., 1.);
}

float4 to_device_position(float2 unit_vertex, Bounds_ScaledPixels bounds,
                          constant Size_DevicePixels *input_viewport_size) {
  float2 position =
      unit_vertex * float2(bounds.size.width, bounds.size.height) +
      float2(bounds.origin.x, bounds.origin.y);
  return to_device_position_impl(position, input_viewport_size);
}

// Must match `gpui::quad_depth`.
float quad_depth(uint quad_id) {
  return saturate(float(quad_id + 1u) * (1.0 / 65535.0));
}

float4 to_device_position_transformed(float2 unit_vertex, Bounds_ScaledPixels bounds,
                          TransformationMatrix transformation,
                          constant Size_DevicePixels *input_viewport_size) {
  float2 position =
      unit_vertex * float2(bounds.size.width, bounds.size.height) +
      float2(bounds.origin.x, bounds.origin.y);

  // Apply the transformation matrix to the position via matrix multiplication.
  float2 transformed_position = float2(0, 0);
  transformed_position[0] = position[0] * transformation.rotation_scale[0][0] + position[1] * transformation.rotation_scale[0][1];
  transformed_position[1] = position[0] * transformation.rotation_scale[1][0] + position[1] * transformation.rotation_scale[1][1];

  // Add in the translation component of the transformation matrix.
  transformed_position[0] += transformation.translation[0];
  transformed_position[1] += transformation.translation[1];

  float2 viewport_size = float2((float)input_viewport_size->width,
                                (float)input_viewport_size->height);
  float2 device_position =
      transformed_position / viewport_size * float2(2., -2.) + float2(-1., 1.);
  return float4(device_position, 0., 1.);
}


float2 to_tile_position(float2 unit_vertex, AtlasTile tile,
                        constant Size_DevicePixels *atlas_size) {
  float2 tile_origin = float2(tile.bounds.origin.x, tile.bounds.origin.y);
  float2 tile_size = float2(tile.bounds.size.width, tile.bounds.size.height);
  return (tile_origin + unit_vertex * tile_size) /
         float2((float)atlas_size->width, (float)atlas_size->height);
}

// Selects corner radius based on quadrant.
float pick_corner_radius(float2 center_to_point, Corners_ScaledPixels corner_radii) {
  if (center_to_point.x < 0.) {
    if (center_to_point.y < 0.) {
      return corner_radii.top_left;
    } else {
      return corner_radii.bottom_left;
    }
  } else {
    if (center_to_point.y < 0.) {
      return corner_radii.top_right;
    } else {
      return corner_radii.bottom_right;
    }
  }
}

// Signed distance of the point to the quad's border - positive outside the
// border, and negative inside.
float quad_sdf(float2 point, Bounds_ScaledPixels bounds,
               Corners_ScaledPixels corner_radii) {
    float2 half_size = float2(bounds.size.width, bounds.size.height) / 2.0;
    float2 center = float2(bounds.origin.x, bounds.origin.y) + half_size;
    float2 center_to_point = point - center;
    float corner_radius = pick_corner_radius(center_to_point, corner_radii);
    float2 corner_to_point = fabs(center_to_point) - half_size;
    float2 corner_center_to_point = corner_to_point + corner_radius;
    return quad_sdf_impl(corner_center_to_point, corner_radius);
}

// Implementation of quad signed distance field
float quad_sdf_impl(float2 corner_center_to_point, float corner_radius) {
    if (corner_radius == 0.0) {
        // Fast path for unrounded corners
        return max(corner_center_to_point.x, corner_center_to_point.y);
    } else {
        // Signed distance of the point from a quad that is inset by corner_radius
        // It is negative inside this quad, and positive outside
        float signed_distance_to_inset_quad =
            // 0 inside the inset quad, and positive outside
            length(max(float2(0.0), corner_center_to_point)) +
            // 0 outside the inset quad, and negative inside
            min(0.0, max(corner_center_to_point.x, corner_center_to_point.y));

        return signed_distance_to_inset_quad - corner_radius;
    }
}

// Moves a unit quad's corner inside `mask`, so that what the mask clips is
// never rasterized.
ClippedVertex clip_to_mask(float2 unit_vertex, Bounds_ScaledPixels bounds,
                           Bounds_ScaledPixels mask) {
  float2 bounds_origin = float2(bounds.origin.x, bounds.origin.y);
  float2 bounds_size = float2(bounds.size.width, bounds.size.height);
  float2 mask_origin = float2(mask.origin.x, mask.origin.y);
  float2 mask_size = float2(mask.size.width, mask.size.height);
  float2 origin = max(bounds_origin, mask_origin);
  float2 corner = min(bounds_origin + bounds_size, mask_origin + mask_size);
  float2 size = max(corner - origin, float2(0.));

  ClippedVertex result;
  result.position = origin + unit_vertex * size;
  result.unit_vertex = (result.position - bounds_origin) / bounds_size;
  return result;
}

float4 distance_from_clip_rect_transformed(float2 unit_vertex, Bounds_ScaledPixels bounds,
                               Bounds_ScaledPixels clip_bounds, TransformationMatrix transformation) {
  float2 position =
      unit_vertex * float2(bounds.size.width, bounds.size.height) +
      float2(bounds.origin.x, bounds.origin.y);
  float2 transformed_position = float2(0, 0);
  transformed_position[0] = position[0] * transformation.rotation_scale[0][0] + position[1] * transformation.rotation_scale[0][1];
  transformed_position[1] = position[0] * transformation.rotation_scale[1][0] + position[1] * transformation.rotation_scale[1][1];
  transformed_position[0] += transformation.translation[0];
  transformed_position[1] += transformation.translation[1];

  return float4(transformed_position.x - clip_bounds.origin.x,
                clip_bounds.origin.x + clip_bounds.size.width - transformed_position.x,
                transformed_position.y - clip_bounds.origin.y,
                clip_bounds.origin.y + clip_bounds.size.height - transformed_position.y);
}

float4 over(float4 below, float4 above) {
  float4 result;
  float alpha = above.a + below.a * (1.0 - above.a);
  result.rgb =
      (above.rgb * above.a + below.rgb * below.a * (1.0 - above.a)) / alpha;
  result.a = alpha;
  return result;
}

GradientColor prepare_fill_color(uint tag, uint color_space, Hsla solid,
                                     Hsla color0, Hsla color1) {
  GradientColor out;
  if (tag == 0 || tag == 2 || tag == 3 || tag == 4) {
    out.solid = hsla_to_rgba(solid);
  } else if (tag == 1) {
    out.color0 = hsla_to_rgba(color0);
    out.color1 = hsla_to_rgba(color1);

    // Prepare color space in vertex for avoid conversion
    // in fragment shader for performance reasons
    if (color_space == 1) {
      // Oklab
      out.color0 = srgb_to_oklab(out.color0);
      out.color1 = srgb_to_oklab(out.color1);
    }
  }

  return out;
}

float2x2 rotate2d(float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return float2x2(c, -s, s, c);
}

float4 fill_color(Background background,
                      float2 position,
                      Bounds_ScaledPixels bounds,
                      float4 solid_color, float4 color0, float4 color1) {
  float4 color;

  switch (background.tag) {
    case 0:
      color = solid_color;
      break;
    case 1: {
      // -90 degrees to match the CSS gradient angle.
      float gradient_angle = background.gradient_angle_or_pattern_height;
      float radians = (fmod(gradient_angle, 360.0) - 90.0) * (M_PI_F / 180.0);
      float2 direction = float2(cos(radians), sin(radians));

      // Expand the short side to be the same as the long side
      if (bounds.size.width > bounds.size.height) {
          direction.y *= bounds.size.height / bounds.size.width;
      } else {
          direction.x *=  bounds.size.width / bounds.size.height;
      }

      // Get the t value for the linear gradient with the color stop percentages.
      float2 half_size = float2(bounds.size.width, bounds.size.height) / 2.;
      float2 center = float2(bounds.origin.x, bounds.origin.y) + half_size;
      float2 center_to_point = position - center;
      float t = dot(center_to_point, direction) / length(direction);
      // Check the direction to determine whether to use x or y
      if (abs(direction.x) > abs(direction.y)) {
          t = (t + half_size.x) / bounds.size.width;
      } else {
          t = (t + half_size.y) / bounds.size.height;
      }

      // Adjust t based on the stop percentages
      t = (t - background.colors[0].percentage)
        / (background.colors[1].percentage
        - background.colors[0].percentage);
      t = clamp(t, 0.0, 1.0);

      switch (background.color_space) {
        case 0:
          color = mix(color0, color1, t);
          break;
        case 1: {
          float4 oklab_color = mix(color0, color1, t);
          color = oklab_to_srgb(oklab_color);
          break;
        }
      }

      // Dither to reduce banding in gradients (especially dark/alpha).
      // Triangular-distributed noise breaks up 8-bit quantization steps.
      // ±2/255 for RGB (enough for dark-on-dark compositing),
      // ±3/255 for alpha (needs more because alpha × dark color = tiny steps).
      {
        float2 seed = position * 0.6180339887; // golden ratio spread
        float r1 = fract(sin(dot(seed, float2(12.9898, 78.233))) * 43758.5453);
        float r2 = fract(sin(dot(seed, float2(39.3460, 11.135))) * 24634.6345);
        float tri = r1 + r2 - 1.0; // triangular PDF, range [-1, +1]
        color.rgb += tri * 2.0 / 255.0;
        color.a   += tri * 3.0 / 255.0;
      }

      break;
    }
    case 2: {
        float gradient_angle_or_pattern_height = background.gradient_angle_or_pattern_height;
        float pattern_width = (gradient_angle_or_pattern_height / 65535.0f) / 255.0f;
        float pattern_interval = fmod(gradient_angle_or_pattern_height, 65535.0f) / 255.0f;
        float pattern_height = pattern_width + pattern_interval;
        float stripe_angle = M_PI_F / 4.0;
        float pattern_period = pattern_height * sin(stripe_angle);
        float2x2 rotation = rotate2d(stripe_angle);
        float2 relative_position = position - float2(bounds.origin.x, bounds.origin.y);
        float2 rotated_point = rotation * relative_position;
        float pattern = fmod(rotated_point.x, pattern_period);
        float distance = min(pattern, pattern_period - pattern) - pattern_period * (pattern_width / pattern_height) /  2.0f;
        color = solid_color;
        color.a *= saturate(0.5 - distance);
        break;
    }
    case 3: {
        // checkerboard
        float size = background.gradient_angle_or_pattern_height;
        float2 relative_position = position - float2(bounds.origin.x, bounds.origin.y);

        float x_index = floor(relative_position.x / size);
        float y_index = floor(relative_position.y / size);
        float should_be_colored = fmod(x_index + y_index, 2.0);

        color = solid_color;
        color.a *= saturate(should_be_colored);
        break;
    }
    case 4: {
        float stripe_width = max(background.colors[0].percentage, 0.0);
        float stripe_interval = max(background.colors[1].percentage, 0.0);
        float pattern_period = max(stripe_width + stripe_interval, 1.0);
        float2 relative_position = position - float2(bounds.origin.x, bounds.origin.y);
        float pattern = fmod(relative_position.y, pattern_period);
        float distance =
            min(pattern, pattern_period - pattern) - min(stripe_width, pattern_period) * 0.5;
        color = solid_color;
        color.a *= saturate(0.5 - distance);
        break;
    }
  }

  return color;
}

// How far inside a rounded clip's bounds a point must be for the clip to
// cover it fully: past its largest corner and the antialiasing ramp.
float rounded_clip_interior_inset(RoundedClipNode mask) {
  float radius = max(max(mask.corner_radii.top_left, mask.corner_radii.top_right),
                     max(mask.corner_radii.bottom_right, mask.corner_radii.bottom_left));
  return max(radius, 0.5) + 0.25;
}

float rounded_clip_coverage(uint index, float2 position, constant RoundedClipNode *masks) {
  float coverage = 1.0;
  while (index != 0) {
    RoundedClipNode mask = masks[index - 1];
    // A point farther inside every edge than any corner reaches, and than
    // the antialiasing ramp, is fully covered: its distance field is at most
    // -0.5. Most of a clipped card's pixels are, so they skip the field.
    float inset = rounded_clip_interior_inset(mask);
    bool interior = coordinates_are_precise(mask.bounds) &&
                    position.x > mask.bounds.origin.x + inset &&
                    position.y > mask.bounds.origin.y + inset &&
                    position.x < mask.bounds.origin.x + mask.bounds.size.width - inset &&
                    position.y < mask.bounds.origin.y + mask.bounds.size.height - inset;
    if (!interior) {
      coverage = min(coverage, saturate(0.5 - quad_sdf(position, mask.bounds, mask.corner_radii)));
    }
    index = mask.parent;
  }
  return coverage;
}
