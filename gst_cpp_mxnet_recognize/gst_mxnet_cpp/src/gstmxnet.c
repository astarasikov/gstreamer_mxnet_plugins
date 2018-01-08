/*
 * GStreamer plugin for image recognition via MxNet Copyright (C) 2017
 * Alexander Tarasikov <alexander.tarasikov@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Alternatively, the contents of this file may be used under the GNU Lesser
 * General Public License Version 2.1 (the "LGPL"), in which case the
 * following provisions apply instead of the ones mentioned above:
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Library General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Library General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 */

/**
 * SECTION:element-mxnet
 *
 * FIXME:Describe mxnet here.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! mxnet ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <gst/gst.h>
#include <gst/gstsample.h>

#include "gstmxnet.h"

GST_DEBUG_CATEGORY_STATIC(gst_mxnet_debug);
#define GST_CAT_DEFAULT gst_mxnet_debug

/* Filter signals and args */
enum {
    /* FILL ME */
    LAST_SIGNAL
};

enum {
    PROP_0,
    PROP_SILENT
};

/*
 * the capabilities of the inputs and outputs.
 *
 * describe the real formats here.
 */
static GstStaticPadTemplate sink_factory =
GST_STATIC_PAD_TEMPLATE("sink",
                        GST_PAD_SINK,
                        GST_PAD_ALWAYS,
                        GST_STATIC_CAPS("ANY")
                        );

static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE("src",
                                                                  GST_PAD_SRC,
                                                                  GST_PAD_ALWAYS,
                                                                  GST_STATIC_CAPS("ANY")
                                                                  );

#define gst_mxnet_parent_class parent_class
G_DEFINE_TYPE(Gstmxnet, gst_mxnet, GST_TYPE_BASE_TRANSFORM);

static void	gst_mxnet_set_property(GObject * object, guint prop_id,
                                   const		GValue * value, GParamSpec * pspec);
static void	gst_mxnet_get_property(GObject * object, guint prop_id,
                                   GValue *	value, GParamSpec * pspec);

static		GstFlowReturn
gst_mxnet_template_transform_ip(GstBaseTransform * base, GstBuffer * outbuf);

/* GObject vmethod implementations */

/* initialize the mxnet's class */
static void
gst_mxnet_class_init(GstmxnetClass * klass)
{
    GObjectClass   *gobject_class;
    GstElementClass *gstelement_class;
    
    gobject_class = (GObjectClass *) klass;
    gstelement_class = (GstElementClass *) klass;
    
    gobject_class->set_property = gst_mxnet_set_property;
    gobject_class->get_property = gst_mxnet_get_property;
    
    g_object_class_install_property(gobject_class, PROP_SILENT,
                                    g_param_spec_boolean("silent", "Silent", "Produce verbose output ?",
                                                         FALSE, G_PARAM_READWRITE));
    
    gst_element_class_set_details_simple(gstelement_class,
                                         "mxnet",
                                         "FIXME:Generic",
                                         "FIXME:Generic Template Element",
                                         " <<user@hostname.org>>");
    
    gst_element_class_add_pad_template(gstelement_class,
                                       gst_static_pad_template_get(&src_factory));
    gst_element_class_add_pad_template(gstelement_class,
                                       gst_static_pad_template_get(&sink_factory));
    
    GST_BASE_TRANSFORM_CLASS(klass)->transform_ip = GST_DEBUG_FUNCPTR(gst_mxnet_template_transform_ip);
}

/*
 * initialize the new element instantiate pads and add them to element set
 * pad calback functions initialize instance structure
 */
static void
gst_mxnet_init(Gstmxnet * filter)
{
    filter->silent = FALSE;
}

static void
gst_mxnet_set_property(GObject * object, guint prop_id,
                       const GValue * value, GParamSpec * pspec)
{
    Gstmxnet       *filter = GST_MXNET(object);
    
    switch (prop_id) {
        case PROP_SILENT:
            filter->silent = g_value_get_boolean(value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void
gst_mxnet_get_property(GObject * object, guint prop_id,
                       GValue * value, GParamSpec * pspec)
{
    Gstmxnet       *filter = GST_MXNET(object);
    
    switch (prop_id) {
        case PROP_SILENT:
            g_value_set_boolean(value, filter->silent);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

/* GstBaseTransform vmethod implementations */

#define XCHECK(cond) do { \
	if (!(cond)) { \
		g_print("%s: failed to check (%s)\n", __func__, #cond); \
		goto fail; \
	} \
} while (0)

/*
 * this function does the actual processing
 */
static		GstFlowReturn
gst_mxnet_template_transform_ip(GstBaseTransform * base, GstBuffer * outbuf)
{
    Gstmxnet       *filter = GST_MXNET(base);
	GstMapInfo info = {};
	gint width = 0, height = 0;
	GstCaps *buffer_caps = NULL;
	GstStructure* structure = NULL;
	GstStructure *pool_config = NULL;
	size_t i;

	//https://github.com/opencv/opencv/blob/master/modules/videoio/src/cap_gstreamer.cpp
    
    if (GST_CLOCK_TIME_IS_VALID(GST_BUFFER_TIMESTAMP(outbuf))) {
        gst_object_sync_values(GST_OBJECT(filter), GST_BUFFER_TIMESTAMP(outbuf));
	}

	XCHECK(NULL != outbuf->pool);
	XCHECK(gst_buffer_map(outbuf, &info, GST_MAP_WRITE));
	XCHECK(pool_config = gst_buffer_pool_get_config(outbuf->pool));
	XCHECK(buffer_caps = g_value_get_boxed(gst_structure_get_value(pool_config, "caps")));
	XCHECK(GST_IS_CAPS(buffer_caps));

	XCHECK(gst_caps_get_size(buffer_caps) >= 1);
	XCHECK(structure = gst_caps_get_structure(buffer_caps, 0));

	XCHECK(gst_structure_get_int(structure, "width", &width));
	XCHECK(gst_structure_get_int(structure, "height", &height));

#if 0
	//for testing without MxNet
	for (i = 0; i < info.size / 8; i++) {
		((unsigned char*)info.data)[i] ^= 0xff;
	}

	static int frame = 0;
	if (++frame == 10)
#endif
	{
		extern int gst_mxnet_process_frame(void *data, unsigned width, unsigned height);
		g_print("%s: calling into MxNet CPP plugin\n", __func__);
		gst_mxnet_process_frame(info.data, width, height);
	}

	gst_buffer_unmap(outbuf, &info);

fail:
	if (pool_config) {
		gst_structure_free(pool_config);
	}

    return GST_FLOW_OK;
}


/*
 * entry point to initialize the plug-in initialize the plug-in itself
 * register the element factories and other features
 */
static		gboolean
mxnet_init(GstPlugin * mxnet)
{
    /*
     * debug category for fltering log messages
     * 
     * exchange the string 'Template mxnet' with your description
     */
    GST_DEBUG_CATEGORY_INIT(gst_mxnet_debug, "mxnet",
                            0, "Template mxnet");
    
    return gst_element_register(mxnet, "mxnet", GST_RANK_NONE,
                                GST_TYPE_MXNET);
}

/*
 * PACKAGE: this is usually set by autotools depending on some _INIT macro in
 * configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "gstmxnet"
#endif

/*
 * gstreamer looks for this structure to register mxnets
 * 
 * exchange the string 'Template mxnet' with your mxnet description
 */
GST_PLUGIN_DEFINE(
                  GST_VERSION_MAJOR,
                  GST_VERSION_MINOR,
                  mxnet,
                  "MxNet image recognition",
                  mxnet_init,
                  VERSION,
                  "LGPL",
                  "GStreamer",
                  "http://gstreamer.net/"
                  )
