CREATE TABLE ZMTCHANNEL ( Z_PK INTEGER PRIMARY KEY, Z_ENT INTEGER, Z_OPT INTEGER, ZSTOREID INTEGER, ZSUBSCRIPTIONACTIVE INTEGER, ZAVAILABLESHOWCOUNT INTEGER, ZLASTPERSONALIZEDREQUESTDATE TIMESTAMP, ZLOGOIMAGEHEIGHT FLOAT, ZLOGOIMAGEWIDTH FLOAT, ZSUBSCRIPTIONENABLEDDATE TIMESTAMP, ZARTWORKURL VARCHAR, ZBACKGROUNDCOLOR VARCHAR, ZDISPLAYTYPE VARCHAR, ZLOGOIMAGEURL VARCHAR, ZNAME VARCHAR, ZUBERBACKGROUNDIMAGEURL VARCHAR, ZUBERBACKGROUNDJOECOLOR VARCHAR, ZURL VARCHAR, ZPODCASTUUIDS BLOB , ZSUBSCRIPTIONNAME VARCHAR, ZSHOWCOUNT INTEGER, ZFOLLOWEDSHOWCOUNT INTEGER, ZSUBSCRIPTIONOFFERAPPTYPE VARCHAR, ZCONNECTEDSUBSCRIPTIONTYPE VARCHAR, ZINTEREST INTEGER);
CREATE TABLE IF NOT EXISTS "Z_3PLAYLISTS" ( Z_3EPISODES1 INTEGER, Z_6PLAYLISTS INTEGER, Z_FOK_3EPISODES1 INTEGER, PRIMARY KEY (Z_3EPISODES1, Z_6PLAYLISTS) );
CREATE TABLE IF NOT EXISTS "Z_3SETTINGS" ( Z_3EPISODES INTEGER, Z_9SETTINGS INTEGER, Z_FOK_3EPISODES INTEGER, PRIMARY KEY (Z_3EPISODES, Z_9SETTINGS) );
CREATE TABLE IF NOT EXISTS "Z_3ADDEDTOPLAYLISTS" ( Z_3ADDEDEPISODES INTEGER, Z_6ADDEDTOPLAYLISTS INTEGER, PRIMARY KEY (Z_3ADDEDEPISODES, Z_6ADDEDTOPLAYLISTS) );
CREATE TABLE ZMTOFFLINEKEYDATA ( Z_PK INTEGER PRIMARY KEY, Z_ENT INTEGER, Z_OPT INTEGER, ZPENDINGDELETION INTEGER, ZSECUREINVALIDATIONDSID INTEGER, ZSTORETRACKID INTEGER, ZEXPIRATIONDATE TIMESTAMP, ZLASTRENEWEDDATE TIMESTAMP, ZKEYURI VARCHAR, ZUUID VARCHAR, ZDATA BLOB );
CREATE TABLE IF NOT EXISTS "Z_6PODCASTS" ( Z_6PLAYLISTS1 INTEGER, Z_7PODCASTS1 INTEGER, Z_FOK_7PODCASTS INTEGER, PRIMARY KEY (Z_6PLAYLISTS1, Z_7PODCASTS1) );
CREATE TABLE ZMTPODCAST ( Z_PK INTEGER PRIMARY KEY, Z_ENT INTEGER, Z_OPT INTEGER, ZAUTODOWNLOAD INTEGER, ZAUTODOWNLOADENABLED INTEGER, ZAUTODOWNLOADTYPE INTEGER, ZCHANNELSTOREID INTEGER, ZCONSECUTIVEFEEDFETCHERRORS INTEGER, ZDARKCOUNT INTEGER, ZDARKCOUNTLOCAL INTEGER, ZDELETEPLAYEDEPISODES INTEGER, ZDOWNLOADEDEPISODESCOUNT INTEGER, ZEPISODELIMIT INTEGER, ZFLAGS INTEGER, ZHIDDEN INTEGER, ZHIDESPLAYEDEPISODES INTEGER, ZKEEPEPISODES INTEGER, ZLIBRARYEPISODESCOUNT INTEGER, ZNOTIFICATIONS INTEGER, ZOFFERTYPES INTEGER, ZORPHANEDFROMCLOUD INTEGER, ZPODCASTPID INTEGER, ZSAVEDEPISODESCOUNT INTEGER, ZSAVEDUNPLAYEDEPISODESCOUNT INTEGER, ZSHOWPLACARDFORORPHANEDFROMCLOUD INTEGER, ZSHOWPLACARDFORREMOVEPLAYEDEPISODES INTEGER, ZSHOWPLACARDFORSAVEDEPISODES INTEGER, ZSHOWTYPESETTING INTEGER, ZSORTORDER INTEGER, ZSTORECOLLECTIONID INTEGER, ZSUBSCRIBED INTEGER, ZUPDATEINTERVAL INTEGER, ZCHANNEL INTEGER, ZSYNCINFO INTEGER, ZADDEDDATE TIMESTAMP, ZDOWNLOADEDDATE TIMESTAMP, ZFEEDCHANGEDDATE TIMESTAMP, ZLASTDATEPLAYED TIMESTAMP, ZLASTFETCHEDDATE TIMESTAMP, ZLASTSTOREEPISODESINFOCHECKDATE TIMESTAMP, ZLASTSTOREPODCASTINFOCHECKDATE TIMESTAMP, ZLASTTOUCHDATE TIMESTAMP, ZMODIFIEDDATE TIMESTAMP, ZUPDATEAVG FLOAT, ZUPDATESTDDEV FLOAT, ZUPDATEDDATE TIMESTAMP, ZARTWORKPRIMARYCOLOR VARCHAR, ZAUTHOR VARCHAR, ZCATEGORY VARCHAR, ZDISPLAYTYPE VARCHAR, ZFEEDURL VARCHAR, ZIMAGEURL VARCHAR, ZITEMDESCRIPTION VARCHAR, ZLOGOIMAGEURL VARCHAR, ZNEXTEPISODEUUID VARCHAR, ZPROVIDER VARCHAR, ZSHOWTYPEINFEED VARCHAR, ZSTORECLEANURL VARCHAR, ZSTORESHORTURL VARCHAR, ZTITLE VARCHAR, ZUBERBACKGROUNDIMAGEURL VARCHAR, ZUBERBACKGROUNDJOECOLOR VARCHAR, ZUPDATEDFEEDURL VARCHAR, ZUUID VARCHAR, ZWEBPAGEURL VARCHAR , ZEPISODEUSERFILTER BLOB, ZIMPLICITFOLLOWSUNKNOWNSYNCPROPERTIES BLOB, ZSHOWSPECIFICUPSELLCOPY VARCHAR, ZLASTDISMISSEDEPISODEUPSELLBANNERDATE TIMESTAMP, ZNEWEPISODESCOUNT INTEGER, ZLASTIMPLICITLYFOLLOWEDDATE TIMESTAMP, ZNEWTRAILERSCOUNT INTEGER, ZDOWNLOADEDUNPLAYEDEPISODESCOUNT INTEGER, ZISIMPLICITLYFOLLOWED INTEGER, ZFEEDUNIQUENESSHASH VARCHAR, ZISHIDDENORIMPLICITLYFOLLOWED INTEGER, ZETAG VARCHAR, ZBOOTSTRAPGENERATION BLOB, ZNEXTSYNCTOKEN VARCHAR, ZARTWORKTEXTSECONDARYCOLOR VARCHAR, ZARTWORKTEXTPRIMARYCOLOR VARCHAR, ZUBERARTWORKTEXTPRIMARYCOLOR VARCHAR, ZUBERARTWORKTEXTSECONDARYCOLOR VARCHAR, ZLASTREMOVEDFROMUPNEXTDATE TIMESTAMP, ZARTWORKTEXTQUATERNARYCOLOR VARCHAR, ZLATESTEPISODEAVAILABILITYTIME TIMESTAMP, ZUBERARTWORKTEXTTERTIARYCOLOR VARCHAR, ZUBERARTWORKTEXTQUATERNARYCOLOR VARCHAR, ZARTWORKTEXTTERTIARYCOLOR VARCHAR, ZARTWORKTEMPLATEURL VARCHAR, ZLATESTEXITFROMDARKDOWNLOADS TIMESTAMP, ZLASTUNFOLLOWEDDATE TIMESTAMP, ZINTEREST INTEGER, ZPRIMARYCATEGORY INTEGER);
CREATE TABLE ZMTPODCASTCATEGORY ( Z_PK INTEGER PRIMARY KEY, Z_ENT INTEGER, Z_OPT INTEGER, ZCATEGORYUUID VARCHAR, ZPODCASTUUID VARCHAR );
CREATE TABLE ZMTPODCASTPLAYLISTSETTINGS ( Z_PK INTEGER PRIMARY KEY, Z_ENT INTEGER, Z_OPT INTEGER, ZDOWNLOADED INTEGER, ZEPISODESTOSHOW INTEGER, ZFLAGS INTEGER, ZMEDIATYPE INTEGER, ZPLAYORDER INTEGER, ZSHOWPLAYEDEPISODES INTEGER, ZSORTORDER INTEGER, ZVISIBLE INTEGER, ZPLAYLIST INTEGER, ZPLAYLISTIFDEFAULT INTEGER, ZPODCAST INTEGER, Z_FOK_PLAYLIST INTEGER, ZUUID VARCHAR , ZLATESTEPISODEAVAILABILITYDATE TIMESTAMP, ZEPISODECOUNT INTEGER, ZEARLIESTEPISODEAVAILABILITYDATE TIMESTAMP);
CREATE TABLE ZMTSYNCINFO ( Z_PK INTEGER PRIMARY KEY, Z_ENT INTEGER, Z_OPT INTEGER, ZARTWORKUPDATEREVISION INTEGER, ZENTITYTYPE INTEGER, ZINSERTIONREVISION INTEGER, ZSYNCID INTEGER, ZSYNCABILITY INTEGER, ZUPDATEREVISION INTEGER, ZEPISODE INTEGER, ZPLAYLIST INTEGER, ZPODCAST INTEGER, ZUUID VARCHAR );
CREATE TABLE ZMTTHEME ( Z_PK INTEGER PRIMARY KEY, Z_ENT INTEGER, Z_OPT INTEGER, ZISBACKGROUNDLIGHT INTEGER, ZUUID VARCHAR, ZBACKGROUNDCOLOR BLOB, ZPRIMARYTEXTCOLOR BLOB, ZSECONDARYTEXTCOLOR BLOB );
CREATE TABLE ZMTUPPMETADATA ( Z_PK INTEGER PRIMARY KEY, Z_ENT INTEGER, Z_OPT INTEGER, ZHASBEENPLAYED INTEGER, ZPLAYCOUNT INTEGER, ZBOOKMARKTIME TIMESTAMP, ZTIMESTAMP TIMESTAMP, ZMETADATAIDENTIFIER VARCHAR );
CREATE INDEX Z_2PLAYLISTS_Z_4PLAYLISTS_INDEX ON "Z_3PLAYLISTS" (Z_6PLAYLISTS, Z_3EPISODES1);
CREATE INDEX Z_2SETTINGS_Z_7SETTINGS_INDEX ON "Z_3SETTINGS" (Z_9SETTINGS, Z_3EPISODES);
CREATE INDEX Z_2ADDEDTOPLAYLISTS_Z_4ADDEDTOPLAYLISTS_INDEX ON "Z_3ADDEDTOPLAYLISTS" (Z_6ADDEDTOPLAYLISTS, Z_3ADDEDEPISODES);
CREATE INDEX Z_4PODCASTS_Z_5PODCASTS_INDEX ON "Z_6PODCASTS" (Z_7PODCASTS1, Z_6PLAYLISTS1);
CREATE INDEX ZMTPODCAST_ZCHANNEL_INDEX ON ZMTPODCAST (ZCHANNEL);
CREATE INDEX ZMTPODCAST_ZSYNCINFO_INDEX ON ZMTPODCAST (ZSYNCINFO);
CREATE INDEX Z_MTPodcastCategory_byCategoryUuidIndex ON ZMTPODCASTCATEGORY (ZCATEGORYUUID COLLATE BINARY ASC);
CREATE INDEX Z_MTPodcastCategory_byPodcastUuidIndex ON ZMTPODCASTCATEGORY (ZPODCASTUUID COLLATE BINARY ASC);
CREATE INDEX ZMTPODCASTPLAYLISTSETTINGS_ZPLAYLIST_INDEX ON ZMTPODCASTPLAYLISTSETTINGS (ZPLAYLIST);
CREATE INDEX ZMTPODCASTPLAYLISTSETTINGS_ZPLAYLISTIFDEFAULT_INDEX ON ZMTPODCASTPLAYLISTSETTINGS (ZPLAYLISTIFDEFAULT);
CREATE INDEX ZMTPODCASTPLAYLISTSETTINGS_ZPODCAST_INDEX ON ZMTPODCASTPLAYLISTSETTINGS (ZPODCAST);
CREATE INDEX ZMTSYNCINFO_ZEPISODE_INDEX ON ZMTSYNCINFO (ZEPISODE);
CREATE INDEX ZMTSYNCINFO_ZPLAYLIST_INDEX ON ZMTSYNCINFO (ZPLAYLIST);
CREATE INDEX ZMTSYNCINFO_ZPODCAST_INDEX ON ZMTSYNCINFO (ZPODCAST);
CREATE INDEX Z_MTTheme_byUuidIndex ON ZMTTHEME (ZUUID COLLATE BINARY ASC);
CREATE INDEX Z_MTUPPMetadata_byMetadataIdentifierIndex ON ZMTUPPMETADATA (ZMETADATAIDENTIFIER COLLATE BINARY ASC);
CREATE TABLE Z_METADATA (Z_VERSION INTEGER PRIMARY KEY, Z_UUID VARCHAR(255), Z_PLIST BLOB);
CREATE TABLE Z_MODELCACHE (Z_CONTENT BLOB);
CREATE TABLE ACHANGE ( Z_PK INTEGER PRIMARY KEY, Z_ENT INTEGER, Z_OPT INTEGER, ZCHANGETYPE INTEGER, ZENTITY INTEGER, ZENTITYPK INTEGER, ZTRANSACTIONID INTEGER, ZCOLUMNS BLOB, ZTOMBSTONE0 BLOB, ZTOMBSTONE1 BLOB, ZTOMBSTONE2 BLOB );
CREATE TABLE ATRANSACTION ( Z_PK INTEGER PRIMARY KEY, Z_ENT INTEGER, Z_OPT INTEGER, ZAUTHORTS INTEGER, ZBUNDLEIDTS INTEGER, ZCONTEXTNAMETS INTEGER, ZPROCESSIDTS INTEGER, ZTIMESTAMP FLOAT, ZAUTHOR VARCHAR, ZBUNDLEID VARCHAR, ZCONTEXTNAME VARCHAR, ZPROCESSID VARCHAR, ZQUERYGEN BLOB );
CREATE TABLE ATRANSACTIONSTRING ( Z_PK INTEGER PRIMARY KEY, Z_ENT INTEGER, Z_OPT INTEGER, ZNAME VARCHAR );
CREATE INDEX ACHANGE_ZTRANSACTIONID_INDEX ON ACHANGE (ZTRANSACTIONID);
CREATE INDEX ATRANSACTION_ZAUTHORTS_INDEX ON ATRANSACTION (ZAUTHORTS);
CREATE INDEX ATRANSACTION_ZBUNDLEIDTS_INDEX ON ATRANSACTION (ZBUNDLEIDTS);
CREATE INDEX ATRANSACTION_ZCONTEXTNAMETS_INDEX ON ATRANSACTION (ZCONTEXTNAMETS);
CREATE INDEX ATRANSACTION_ZPROCESSIDTS_INDEX ON ATRANSACTION (ZPROCESSIDTS);
CREATE INDEX Z_TRANSACTION_TransactionAuthorIndex ON ATRANSACTION (ZAUTHOR COLLATE BINARY ASC);
CREATE INDEX Z_TRANSACTION_TransactionTimestampIndex ON ATRANSACTION (ZTIMESTAMP COLLATE BINARY ASC);
CREATE UNIQUE INDEX Z_TRANSACTIONSTRING_UNIQUE_NAME ON ATRANSACTIONSTRING (ZNAME COLLATE BINARY ASC);
CREATE TABLE ZMTPLAYLIST ( Z_PK INTEGER PRIMARY KEY, Z_ENT INTEGER, Z_OPT INTEGER, ZCONTAINERORDER INTEGER, ZDOWNLOADEDCOUNT INTEGER, ZFLAGS INTEGER, ZHIDDEN INTEGER, ZMEDIALIBRARYID INTEGER, ZPARENTMEDIALIBRARYID INTEGER, ZSORTORDER INTEGER, ZUNPLAYEDCOUNT INTEGER, ZDEFAULTSETTINGS INTEGER, ZSYNCINFO INTEGER, ZGENERATEDDATE TIMESTAMP, ZUPDATEDDATE TIMESTAMP, ZPLAYERCURRENTEPISODEUUID VARCHAR, ZTITLE VARCHAR, ZUUID VARCHAR );
CREATE TABLE ZMTEPISODE ( Z_PK INTEGER PRIMARY KEY, Z_ENT INTEGER, Z_OPT INTEGER, ZAUDIO INTEGER, ZBACKCATALOG INTEGER, ZBYTESIZE INTEGER, ZDOWNLOADBEHAVIOR INTEGER, ZELIGIBLEFORAUTODOWNLOAD INTEGER, ZENTITLEMENTSTATE INTEGER, ZEPISODELEVEL INTEGER, ZEPISODENUMBER INTEGER, ZEPISODESHOWTYPESPECIFICLEVEL INTEGER, ZEXPLICIT INTEGER, ZEXTERNALTYPE INTEGER, ZFEEDDELETED INTEGER, ZFLAGS INTEGER, ZHASBEENPLAYED INTEGER, ZIMPORTSOURCE INTEGER, ZISBOOKMARKED INTEGER, ZISBOOKMARKSMIGRATIONRECOVEREDEPISODE INTEGER, ZISFROMITUNESSYNC INTEGER, ZISHIDDEN INTEGER, ZISNEW INTEGER, ZITEMDESCRIPTIONHASHTML INTEGER, ZLISTENNOWEPISODE INTEGER, ZMANUALLYADDED INTEGER, ZMARKASPLAYED INTEGER, ZMETADATADIRTY INTEGER, ZMETADATAFIRSTSYNCELIGIBLE INTEGER, ZPERSISTENTID INTEGER, ZPLAYCOUNT INTEGER, ZPLAYSTATE INTEGER, ZPLAYSTATEMANUALLYSET INTEGER, ZPLAYSTATESOURCE INTEGER, ZSAVED INTEGER, ZSEASONNUMBER INTEGER, ZSENTNOTIFICATION INTEGER, ZSTORETRACKID INTEGER, ZSUPPRESSAUTODOWNLOAD INTEGER, ZTRACKNUM INTEGER, ZUNPLAYEDTAB INTEGER, ZUSERDELETED INTEGER, ZUSEREPISODE INTEGER, ZVIDEO INTEGER, ZVISIBLE INTEGER, ZPODCAST INTEGER, ZSYNCINFO INTEGER, ZARTWORKHEIGHT FLOAT, ZARTWORKWIDTH FLOAT, ZDOWNLOADDATE TIMESTAMP, ZDURATION FLOAT, ZENTITLEDDURATION FLOAT, ZFIRSTTIMEAVAILABLE TIMESTAMP, ZFIRSTTIMEAVAILABLEASFREE TIMESTAMP, ZFIRSTTIMEAVAILABLEASPAID TIMESTAMP, ZFREEDURATION FLOAT, ZIMPORTDATE TIMESTAMP, ZLASTBOOKMARKEDDATE TIMESTAMP, ZLASTDATEPLAYED TIMESTAMP, ZLASTUSERMARKEDASPLAYEDDATE TIMESTAMP, ZMETADATATIMESTAMP TIMESTAMP, ZMODIFIEDDATESCORE FLOAT, ZPLAYSTATELASTMODIFIEDDATE TIMESTAMP, ZPLAYHEAD FLOAT, ZPRICETYPECHANGEDDATE TIMESTAMP, ZPUBDATE TIMESTAMP, ZUPNEXTADDEDDATE TIMESTAMP, ZUPNEXTSCORE FLOAT, ZARTWORKBACKGROUNDCOLOR VARCHAR, ZARTWORKTEMPLATEURL VARCHAR, ZARTWORKTEXTPRIMARYCOLOR VARCHAR, ZARTWORKTEXTQUATERNARYCOLOR VARCHAR, ZARTWORKTEXTSECONDARYCOLOR VARCHAR, ZARTWORKTEXTTERTIARYCOLOR VARCHAR, ZASSETURL VARCHAR, ZAUTHOR VARCHAR, ZCATEGORY VARCHAR, ZCLEANEDTITLE VARCHAR, ZDOWNLOADPATH VARCHAR, ZDRMKEYURI VARCHAR, ZENCLOSUREURL VARCHAR, ZENTITLEDENCLOSUREURL VARCHAR, ZENTITLEDPRICETYPE VARCHAR, ZEPISODETYPE VARCHAR, ZFREEENCLOSUREURL VARCHAR, ZFREEPRICETYPE VARCHAR, ZGUID VARCHAR, ZITEMDESCRIPTION VARCHAR, ZITEMDESCRIPTIONWITHOUTHTML VARCHAR, ZITUNESSUBTITLE VARCHAR, ZITUNESTITLE VARCHAR, ZMETADATAIDENTIFIER VARCHAR, ZPODCASTUUID VARCHAR, ZPRICETYPE VARCHAR, ZTITLE VARCHAR, ZUTI VARCHAR, ZUUID VARCHAR, ZWEBPAGEURL VARCHAR, ZBOOTSTRAPGENERATION BLOB, ZITEMDESCRIPTIONWITHHTML BLOB, ZITEMDESCRIPTIONWITHHTMLDATA BLOB, ZSECURITYSCOPEDASSETDATA BLOB , ZENTITLEDTRANSCRIPTSNIPPET VARCHAR, ZFREETRANSCRIPTSNIPPET VARCHAR, ZENTITLEDTRANSCRIPTPROVIDER VARCHAR, ZFREETRANSCRIPTPROVIDER VARCHAR, ZFREETRANSCRIPTIDENTIFIER VARCHAR, ZTRANSCRIPTIDENTIFIER VARCHAR, ZENTITLEDTRANSCRIPTIDENTIFIER VARCHAR, ZLASTCACHEDELETEPURGE TIMESTAMP);
CREATE TABLE IF NOT EXISTS "Z_3DELETEDFROMPLAYLISTS" ( Z_3DELETEDEPISODES INTEGER, Z_6DELETEDFROMPLAYLISTS INTEGER, PRIMARY KEY (Z_3DELETEDEPISODES, Z_6DELETEDFROMPLAYLISTS) );
CREATE INDEX ZMTEPISODE_ZPODCAST_INDEX ON ZMTEPISODE (ZPODCAST);
CREATE INDEX ZMTEPISODE_ZSYNCINFO_INDEX ON ZMTEPISODE (ZSYNCINFO);
CREATE INDEX Z_2DELETEDFROMPLAYLISTS_Z_4DELETEDFROMPLAYLISTS_INDEX ON "Z_3DELETEDFROMPLAYLISTS" (Z_6DELETEDFROMPLAYLISTS, Z_3DELETEDEPISODES);
CREATE INDEX ZMTPLAYLIST_ZDEFAULTSETTINGS_INDEX ON ZMTPLAYLIST (ZDEFAULTSETTINGS);
CREATE INDEX ZMTPLAYLIST_ZSYNCINFO_INDEX ON ZMTPLAYLIST (ZSYNCINFO);
CREATE INDEX Z_MTPlaylist_byHiddenIndex ON ZMTPLAYLIST (ZHIDDEN COLLATE BINARY ASC);
CREATE INDEX Z_MTPlaylist_byMediaLibraryIdIndex ON ZMTPLAYLIST (ZMEDIALIBRARYID COLLATE BINARY ASC);
CREATE INDEX Z_MTPlaylist_byPlayerCurrentEpisodeUuidIndex ON ZMTPLAYLIST (ZPLAYERCURRENTEPISODEUUID COLLATE BINARY ASC);
CREATE INDEX Z_MTPlaylist_byUuidIndex ON ZMTPLAYLIST (ZUUID COLLATE BINARY ASC);
CREATE TABLE ZMTINTEREST ( Z_PK INTEGER PRIMARY KEY, Z_ENT INTEGER, Z_OPT INTEGER, ZADAMID INTEGER, ZINTERESTVALUE FLOAT, ZLASTUPDATEDDATE TIMESTAMP , ZPODCAST INTEGER, ZCATEGORY INTEGER, ZUNKNOWNSYNCPROPERTIES BLOB, ZCHANNEL INTEGER);
CREATE INDEX Z_2PLAYLISTS_Z_5PLAYLISTS_INDEX ON "Z_3PLAYLISTS" (Z_6PLAYLISTS, Z_3EPISODES1);
CREATE INDEX Z_2DELETEDFROMPLAYLISTS_Z_5DELETEDFROMPLAYLISTS_INDEX ON "Z_3DELETEDFROMPLAYLISTS" (Z_6DELETEDFROMPLAYLISTS, Z_3DELETEDEPISODES);
CREATE INDEX Z_2SETTINGS_Z_8SETTINGS_INDEX ON "Z_3SETTINGS" (Z_9SETTINGS, Z_3EPISODES);
CREATE INDEX Z_2ADDEDTOPLAYLISTS_Z_5ADDEDTOPLAYLISTS_INDEX ON "Z_3ADDEDTOPLAYLISTS" (Z_6ADDEDTOPLAYLISTS, Z_3ADDEDEPISODES);
CREATE INDEX Z_5PODCASTS_Z_6PODCASTS_INDEX ON "Z_6PODCASTS" (Z_7PODCASTS1, Z_6PLAYLISTS1);
CREATE TABLE ZMTCATEGORY ( Z_PK INTEGER PRIMARY KEY, Z_ENT INTEGER, Z_OPT INTEGER, ZADAMID INTEGER, ZARTWORKHEIGHTNUMBER INTEGER, ZARTWORKWIDTHNUMBER INTEGER, ZINTEREST INTEGER, ZARTWORKPRIMARYCOLOR VARCHAR, ZARTWORKTEMPLATEURL VARCHAR, ZCOLOR VARCHAR, ZNAME VARCHAR, ZURL VARCHAR , ZPARENT INTEGER);
CREATE TABLE Z_1PODCASTS ( Z_1CATEGORIES INTEGER, Z_7PODCASTS INTEGER, Z_FOK_1CATEGORIES INTEGER, PRIMARY KEY (Z_1CATEGORIES, Z_7PODCASTS) );
CREATE INDEX ZMTCATEGORY_ZINTEREST_INDEX ON ZMTCATEGORY (ZINTEREST);
CREATE INDEX Z_1PODCASTS_Z_7PODCASTS_INDEX ON Z_1PODCASTS (Z_7PODCASTS, Z_1CATEGORIES);
CREATE INDEX ZMTCHANNEL_ZINTEREST_INDEX ON ZMTCHANNEL (ZINTEREST);
CREATE INDEX Z_MTChannel_byStoreId ON ZMTCHANNEL (ZSTOREID COLLATE BINARY ASC);
CREATE INDEX Z_MTChannel_byName ON ZMTCHANNEL (ZNAME COLLATE BINARY ASC);
CREATE INDEX Z_MTChannel_bySubscriptionActive ON ZMTCHANNEL (ZSUBSCRIPTIONACTIVE COLLATE BINARY ASC);
CREATE INDEX Z_3PLAYLISTS_Z_6PLAYLISTS_INDEX ON Z_3PLAYLISTS (Z_6PLAYLISTS, Z_3EPISODES1);
CREATE INDEX Z_3DELETEDFROMPLAYLISTS_Z_6DELETEDFROMPLAYLISTS_INDEX ON Z_3DELETEDFROMPLAYLISTS (Z_6DELETEDFROMPLAYLISTS, Z_3DELETEDEPISODES);
CREATE INDEX Z_3SETTINGS_Z_9SETTINGS_INDEX ON Z_3SETTINGS (Z_9SETTINGS, Z_3EPISODES);
CREATE INDEX Z_3ADDEDTOPLAYLISTS_Z_6ADDEDTOPLAYLISTS_INDEX ON Z_3ADDEDTOPLAYLISTS (Z_6ADDEDTOPLAYLISTS, Z_3ADDEDEPISODES);
CREATE INDEX ZMTINTEREST_ZCATEGORY_INDEX ON ZMTINTEREST (ZCATEGORY);
CREATE INDEX ZMTINTEREST_ZCHANNEL_INDEX ON ZMTINTEREST (ZCHANNEL);
CREATE INDEX ZMTINTEREST_ZPODCAST_INDEX ON ZMTINTEREST (ZPODCAST);
CREATE INDEX Z_6PODCASTS_Z_7PODCASTS1_INDEX ON Z_6PODCASTS (Z_7PODCASTS1, Z_6PLAYLISTS1);
CREATE INDEX ZMTPODCAST_ZINTEREST_INDEX ON ZMTPODCAST (ZINTEREST);
CREATE TABLE Z_PRIMARYKEY (Z_ENT INTEGER PRIMARY KEY, Z_NAME VARCHAR, Z_SUPER INTEGER, Z_MAX INTEGER);
CREATE INDEX ZMTCATEGORY_ZPARENT_INDEX ON ZMTCATEGORY (ZPARENT);
CREATE INDEX Z_MTEpisode_byAssetURLIndex ON ZMTEPISODE (ZASSETURL COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byEligibleForAutoDownloadIndex ON ZMTEPISODE (ZELIGIBLEFORAUTODOWNLOAD COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byEpisodeLevelIndex ON ZMTEPISODE (ZEPISODELEVEL COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byEpisodeTypeIndex ON ZMTEPISODE (ZEPISODETYPE COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byExplicitIndex ON ZMTEPISODE (ZEXPLICIT COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byFlagsIndex ON ZMTEPISODE (ZFLAGS COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byFromiTunesIndex ON ZMTEPISODE (ZISFROMITUNESSYNC COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byHasBeenPlayedIndex ON ZMTEPISODE (ZHASBEENPLAYED COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byImportDateIndex ON ZMTEPISODE (ZIMPORTDATE COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byIsNewIndex ON ZMTEPISODE (ZISNEW COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byLastDatePlayedIndex ON ZMTEPISODE (ZLASTDATEPLAYED COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byLastUserMarkedAsPlayedDateIndex ON ZMTEPISODE (ZLASTUSERMARKEDASPLAYEDDATE COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byListenNowEpisodeIndex ON ZMTEPISODE (ZLISTENNOWEPISODE COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byMarkAsPlayedIndex ON ZMTEPISODE (ZMARKASPLAYED COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byMetadataDirtyIndex ON ZMTEPISODE (ZMETADATADIRTY COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byMetadataFirstSyncEligibleIndex ON ZMTEPISODE (ZMETADATAFIRSTSYNCELIGIBLE COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byMetadataIdentifierIndex ON ZMTEPISODE (ZMETADATAIDENTIFIER COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byPersistentIDIndex ON ZMTEPISODE (ZPERSISTENTID COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byPlayStateIndex ON ZMTEPISODE (ZPLAYSTATE COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byPlayStateManuallySetIndex ON ZMTEPISODE (ZPLAYSTATEMANUALLYSET COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byPodcastUuidIndex ON ZMTEPISODE (ZPODCASTUUID COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byPubDateIndex ON ZMTEPISODE (ZPUBDATE COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_bySavedIndex ON ZMTEPISODE (ZSAVED COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_bySeasonNumberIndex ON ZMTEPISODE (ZSEASONNUMBER COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byUnplayedTabIndex ON ZMTEPISODE (ZUNPLAYEDTAB COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byUserEpisodeIndex ON ZMTEPISODE (ZUSEREPISODE COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byUuidIndex ON ZMTEPISODE (ZUUID COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byVideoIndex ON ZMTEPISODE (ZVIDEO COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byVisibleIndex ON ZMTEPISODE (ZVISIBLE COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byMetadataTimestamp ON ZMTEPISODE (ZMETADATATIMESTAMP COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byPubDateNonNullAssetURL ON ZMTEPISODE (ZPUBDATE COLLATE BINARY ASC) WHERE ZASSETURL IS NOT NULL;
CREATE INDEX Z_MTEpisode_byLastDatePlayedNonNullAssetURL ON ZMTEPISODE (ZLASTDATEPLAYED COLLATE BINARY ASC) WHERE ZASSETURL IS NOT NULL;
CREATE INDEX Z_MTEpisode_byContentTypeIndex ON ZMTEPISODE (ZPRICETYPE COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byHiddenIndex ON ZMTEPISODE (ZISHIDDEN COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byBookmarked ON ZMTEPISODE (ZISBOOKMARKED COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_upNextShelfCompoundIndex ON ZMTEPISODE (ZISHIDDEN COLLATE BINARY ASC, ZLISTENNOWEPISODE COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_podcastUuidCompoundIndex ON ZMTEPISODE (ZISHIDDEN COLLATE BINARY ASC, ZPODCASTUUID COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_lastPlayedDateCompoundIndex ON ZMTEPISODE (ZISHIDDEN COLLATE BINARY ASC, ZLASTDATEPLAYED COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_lastPlayedDateReverseCompoundIndex ON ZMTEPISODE (ZLASTDATEPLAYED COLLATE BINARY ASC, ZISHIDDEN COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_userEpisodeCompoundIndex ON ZMTEPISODE (ZUSEREPISODE COLLATE BINARY ASC, ZISHIDDEN COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_bookmarkedCompoundIndex ON ZMTEPISODE (ZISBOOKMARKED COLLATE BINARY ASC, ZISHIDDEN COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_flagsCompoundIndex ON ZMTEPISODE (ZFLAGS COLLATE BINARY ASC, ZISHIDDEN COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byStoreTrackIdIndex ON ZMTEPISODE (ZSTORETRACKID COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byGuidIndex ON ZMTEPISODE (ZGUID COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byFirstTimeAvailableIndex ON ZMTEPISODE (ZFIRSTTIMEAVAILABLE COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byShowUuidAndPubDateIndex ON ZMTEPISODE (ZPODCASTUUID COLLATE BINARY ASC, ZPUBDATE COLLATE BINARY ASC, ZEPISODELEVEL COLLATE BINARY DESC, ZTITLE COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byPubDateMulti ON ZMTEPISODE (ZPODCASTUUID COLLATE BINARY ASC, ZPUBDATE COLLATE BINARY ASC, ZEPISODENUMBER COLLATE BINARY ASC, ZEPISODELEVEL COLLATE BINARY DESC, ZTITLE COLLATE BINARY DESC);
CREATE INDEX Z_MTEpisode_byLastDatePlayedMulti ON ZMTEPISODE (ZPODCASTUUID COLLATE BINARY ASC, ZLASTDATEPLAYED COLLATE BINARY ASC, ZPLAYSTATE COLLATE BINARY ASC, ZPUBDATE COLLATE BINARY DESC);
CREATE INDEX Z_MTEpisode_byEpisodeLevelMulti ON ZMTEPISODE (ZPODCASTUUID COLLATE BINARY ASC, ZPUBDATE COLLATE BINARY DESC, ZEPISODENUMBER COLLATE BINARY DESC, ZEPISODELEVEL COLLATE BINARY ASC);
CREATE INDEX Z_MTEpisode_byDownloadBehavior ON ZMTEPISODE (ZDOWNLOADBEHAVIOR COLLATE BINARY ASC);
CREATE INDEX ZMTPODCAST_ZPRIMARYCATEGORY_INDEX ON ZMTPODCAST (ZPRIMARYCATEGORY);
CREATE INDEX Z_MTPodcast_byDarkCountIndex ON ZMTPODCAST (ZDARKCOUNT COLLATE BINARY ASC);
CREATE INDEX Z_MTPodcast_byFeedChangedDateIndex ON ZMTPODCAST (ZFEEDCHANGEDDATE COLLATE BINARY ASC);
CREATE INDEX Z_MTPodcast_byFeedURLIndex ON ZMTPODCAST (ZFEEDURL COLLATE BINARY ASC);
CREATE INDEX Z_MTPodcast_byLastDatePlayedIndex ON ZMTPODCAST (ZLASTDATEPLAYED COLLATE BINARY ASC);
CREATE INDEX Z_MTPodcast_byModifiedDateIndex ON ZMTPODCAST (ZMODIFIEDDATE COLLATE BINARY ASC);
CREATE INDEX Z_MTPodcast_byShowTypeInFeedIndex ON ZMTPODCAST (ZSHOWTYPEINFEED COLLATE BINARY ASC);
CREATE INDEX Z_MTPodcast_byUpdatedFeedURLIndex ON ZMTPODCAST (ZUPDATEDFEEDURL COLLATE BINARY ASC);
CREATE INDEX Z_MTPodcast_byUuidIndex ON ZMTPODCAST (ZUUID COLLATE BINARY ASC);
CREATE INDEX Z_MTPodcast_byDisplayTypeIndex ON ZMTPODCAST (ZDISPLAYTYPE COLLATE BINARY ASC);
CREATE INDEX Z_MTPodcast_byOfferTypesIndex ON ZMTPODCAST (ZOFFERTYPES COLLATE BINARY ASC);
CREATE TRIGGER Z_DA_ZMTPODCAST_MTChannel_followedShowCount_PARAMETER_INSERT AFTER INSERT ON ZMTPODCAST FOR EACH ROW WHEN (NEW.ZCHANNEL NOT NULL) BEGIN UPDATE ZMTCHANNEL SET ZFOLLOWEDSHOWCOUNT = IFNULL(ZFOLLOWEDSHOWCOUNT, 0) + IFNULL(NEW.ZSUBSCRIBED, 0) WHERE Z_PK = NEW.ZCHANNEL; SELECT NSCoreDataDATriggerUpdatedAffectedObjectValue('ZMTCHANNEL', Z_ENT, Z_PK, 'followedShowCount', ZFOLLOWEDSHOWCOUNT) FROM ZMTCHANNEL WHERE Z_PK = NEW.ZCHANNEL; END;
CREATE TRIGGER Z_DA_ZMTPODCAST_MTChannel_followedShowCount_PARAMETER_UPDATE_INCREMENT AFTER UPDATE OF ZCHANNEL, ZSUBSCRIBED ON ZMTPODCAST FOR EACH ROW WHEN (NEW.ZCHANNEL NOT NULL) BEGIN UPDATE ZMTCHANNEL SET ZFOLLOWEDSHOWCOUNT = ZFOLLOWEDSHOWCOUNT + IFNULL(NEW.ZSUBSCRIBED, 0) WHERE Z_PK = NEW.ZCHANNEL; SELECT NSCoreDataDATriggerUpdatedAffectedObjectValue('ZMTCHANNEL', Z_ENT, Z_PK, 'followedShowCount', ZFOLLOWEDSHOWCOUNT) FROM ZMTCHANNEL WHERE Z_PK = NEW.ZCHANNEL; END;
CREATE TRIGGER Z_DA_ZMTPODCAST_MTChannel_followedShowCount_PARAMETER_UPDATE_DECREMENT AFTER UPDATE OF ZCHANNEL, ZSUBSCRIBED ON ZMTPODCAST FOR EACH ROW WHEN (OLD.ZCHANNEL NOT NULL) BEGIN UPDATE ZMTCHANNEL SET ZFOLLOWEDSHOWCOUNT = ZFOLLOWEDSHOWCOUNT - IFNULL(OLD.ZSUBSCRIBED, 0) WHERE Z_PK = OLD.ZCHANNEL; SELECT NSCoreDataDATriggerUpdatedAffectedObjectValue('ZMTCHANNEL', Z_ENT, Z_PK, 'followedShowCount', ZFOLLOWEDSHOWCOUNT) FROM ZMTCHANNEL WHERE Z_PK = OLD.ZCHANNEL; END;
CREATE TRIGGER Z_DA_ZMTPODCAST_MTChannel_followedShowCount_PARAMETER_DELETE AFTER DELETE ON ZMTPODCAST FOR EACH ROW WHEN (OLD.ZCHANNEL NOT NULL) BEGIN UPDATE ZMTCHANNEL SET ZFOLLOWEDSHOWCOUNT = ZFOLLOWEDSHOWCOUNT - IFNULL(OLD.ZSUBSCRIBED, 0) WHERE Z_PK = OLD.ZCHANNEL; SELECT NSCoreDataDATriggerUpdatedAffectedObjectValue('ZMTCHANNEL', Z_ENT, Z_PK, 'followedShowCount', ZFOLLOWEDSHOWCOUNT) FROM ZMTCHANNEL WHERE Z_PK = OLD.ZCHANNEL; END;
CREATE TRIGGER Z_DA_ZMTCHANNEL_MTChannel_followedShowCount_SOURCE_INSERT AFTER INSERT ON ZMTCHANNEL FOR EACH ROW BEGIN UPDATE ZMTCHANNEL SET ZFOLLOWEDSHOWCOUNT = (SELECT IFNULL(SUM(ZSUBSCRIBED), 0) FROM ZMTPODCAST WHERE ZCHANNEL = NEW.Z_PK) WHERE Z_PK = NEW.Z_PK; SELECT NSCoreDataDATriggerInsertUpdatedAffectedObjectValue('ZMTCHANNEL', Z_ENT, Z_PK, 'followedShowCount', ZFOLLOWEDSHOWCOUNT) FROM ZMTCHANNEL WHERE Z_PK = NEW.Z_PK; END;
